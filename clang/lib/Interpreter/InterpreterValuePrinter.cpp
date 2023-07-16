//===--- InterpreterValuePrinter.cpp - Value printing utils -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for in-process value printing in clang-repl.
//
//===----------------------------------------------------------------------===//

#include "IncrementalParser.h"
#include "InterpreterUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/Value.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>

using namespace clang;

static std::string PrintDeclType(const QualType &QT, NamedDecl *D) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  if (QT.hasQualifiers())
    SS << QT.getQualifiers().getAsString() << " ";
  SS << D->getQualifiedNameAsString();
  return Str;
}

static std::string PrintQualType(ASTContext &Ctx, QualType QT) {
  PrintingPolicy Policy(Ctx.getPrintingPolicy());
  // Print the Allocator in STL containers, for instance.
  Policy.SuppressDefaultTemplateArgs = false;
  Policy.SuppressUnwrittenScope = true;
  // Print 'a<b<c> >' rather than 'a<b<c>>'.
  Policy.SplitTemplateClosers = true;

  struct LocalPrintingPolicyRAII {
    ASTContext &Context;
    PrintingPolicy Policy;

    LocalPrintingPolicyRAII(ASTContext &Ctx, PrintingPolicy &PP)
        : Context(Ctx), Policy(Ctx.getPrintingPolicy()) {
      Context.setPrintingPolicy(PP);
    }
    ~LocalPrintingPolicyRAII() { Context.setPrintingPolicy(Policy); }
  } X(Ctx, Policy);

  const QualType NonRefTy = QT.getNonReferenceType();

  if (const auto *TTy = llvm::dyn_cast<TagType>(NonRefTy))
    return PrintDeclType(NonRefTy, TTy->getDecl());

  if (const auto *TRy = dyn_cast<RecordType>(NonRefTy))
    return PrintDeclType(NonRefTy, TRy->getDecl());

  const QualType Canon = NonRefTy.getCanonicalType();

  // FIXME: How a builtin type can be a function pointer type?
  if (Canon->isBuiltinType() && !NonRefTy->isFunctionPointerType() &&
      !NonRefTy->isMemberPointerType())
    return Canon.getAsString(Ctx.getPrintingPolicy());

  if (const auto *TDTy = dyn_cast<TypedefType>(NonRefTy)) {
    // FIXME: TemplateSpecializationType & SubstTemplateTypeParmType checks
    // are predominately to get STL containers to print nicer and might be
    // better handled in GetFullyQualifiedName.
    //
    // std::vector<Type>::iterator is a TemplateSpecializationType
    // std::vector<Type>::value_type is a SubstTemplateTypeParmType
    //
    QualType SSDesugar = TDTy->getLocallyUnqualifiedSingleStepDesugaredType();
    if (llvm::isa<SubstTemplateTypeParmType>(SSDesugar))
      return GetFullTypeName(Ctx, Canon);
    else if (llvm::isa<TemplateSpecializationType>(SSDesugar))
      return GetFullTypeName(Ctx, NonRefTy);
    return PrintDeclType(NonRefTy, TDTy->getDecl());
  }
  return GetFullTypeName(Ctx, NonRefTy);
}

static std::string PrintEnum(const Value &V) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  ASTContext &Ctx = const_cast<ASTContext &>(V.getASTContext());

  QualType DesugaredTy = V.getType().getDesugaredType(Ctx);
  const EnumType *EnumTy = DesugaredTy.getNonReferenceType()->getAs<EnumType>();
  assert(EnumTy && "Fail to cast to enum type");

  EnumDecl *ED = EnumTy->getDecl();
  uint64_t Data = V.getULongLong();
  bool IsFirst = true;
  llvm::APSInt AP = Ctx.MakeIntValue(Data, DesugaredTy);

  for (auto I = ED->enumerator_begin(), E = ED->enumerator_end(); I != E; ++I) {
    if (I->getInitVal() == AP) {
      if (!IsFirst)
        SS << " ? ";
      SS << "(" + I->getQualifiedNameAsString() << ")";
      IsFirst = false;
    }
  }
  llvm::SmallString<64> APStr;
  AP.toString(APStr, /*Radix=*/10);
  SS << " : " << PrintQualType(Ctx, ED->getIntegerType()) << " " << APStr;
  return Str;
}

static std::string PrintFunction(const Value &V, const void *Ptr) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "Function @" << Ptr;

  const FunctionDecl *FD = nullptr;

  auto Decls = V.getASTContext().getTranslationUnitDecl()->decls();
  assert(std::distance(Decls.begin(), Decls.end()) == 1 &&
         "TU should only contain one Decl");
  auto *TLSD = llvm::cast<TopLevelStmtDecl>(*Decls.begin());

  // Get __clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void
  // *OpaqueType, void *Val);
  if (auto *InterfaceCall = llvm::dyn_cast<CallExpr>(TLSD->getStmt())) {
    const auto *Arg = InterfaceCall->getArg(/*Val*/ 3);
    // Get rid of cast nodes.
    while (const CastExpr *CastE = llvm::dyn_cast<CastExpr>(Arg))
      Arg = CastE->getSubExpr();
    if (const DeclRefExpr *DeclRefExp = llvm::dyn_cast<DeclRefExpr>(Arg))
      FD = llvm::dyn_cast<FunctionDecl>(DeclRefExp->getDecl());

    if (FD) {
      SS << '\n';
      const clang::FunctionDecl *FDef;
      if (FD->hasBody(FDef))
        FDef->print(SS);
    }
  }
  return Str;
}

static std::string PrintAddress(const void *Ptr, char Prefix) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  if (!Ptr)
    return Str;
  SS << Prefix << Ptr;
  return Str;
}

// FIXME: Encodings. Handle unprintable characters such as control characters.
static std::string PrintOneChar(char Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  SS << "'" << Val << "'";
  return Str;
}

// Char pointers
// Assumption is this is a string.
// N is limit to prevent endless loop if Ptr is not really a string.
static std::string PrintString(const char *const *Ptr, size_t N = 10000) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  const char *Start = *Ptr;
  if (!Start)
    return "nullptr";

  const char *End = Start + N;
  // If we're gonna do this, better make sure the end is valid too
  // FIXME: getpagesize() & GetSystemInfo().dwPageSize might be better
  static constexpr auto PAGE_SIZE = 1024;
  while (N > 1024) {
    N -= PAGE_SIZE;
    End = Start + N;
  }

  if (*Start == 0)
    return "\"\"";

  // Copy the bytes until we get a null-terminator
  SS << "\"";
  while (Start < End && *Start)
    SS << *Start++;
  SS << "\"";

  return Str;
}

// Build the CallExpr to `PrintValueRuntime`.
static void BuildWrapperBody(Interpreter &Interp, Sema &S, ASTContext &Ctx,
                             FunctionDecl *WrapperFD, QualType QT,
                             const void *ValPtr) {
  Sema::SynthesizedFunctionScope SemaFScope(S, WrapperFD);
  clang::DeclarationName RuntimeCallName;
  if (Ctx.getLangOpts().CPlusPlus)
    RuntimeCallName = S.PP.getIdentifierInfo("PrintValueRuntime");
  else
    RuntimeCallName =
        S.PP.getIdentifierInfo("caas__runtime__PrintValueRuntime");

  clang::LookupResult R(S, RuntimeCallName, SourceLocation(),
                        clang::Sema::LookupOrdinaryName);
  S.LookupName(R, S.getCurScope());

  Expr *OverldExpr = UnresolvedLookupExpr::Create(
      Ctx, /*NamingClass=*/nullptr, NestedNameSpecifierLoc(),
      clang::DeclarationNameInfo(RuntimeCallName, SourceLocation()),
      /*RequiresADL=*/false, R.begin(), R.end(), /*KnownDependent=*/false,
      /*KnownInstantiationDependent=*/false);

  if (const auto *AT = llvm::dyn_cast<AutoType>(QT.getTypePtr())) {
    if (AT->isDeduced())
      QT = AT->getDeducedType().getDesugaredType(Ctx);
  }

  if (const auto *PT = llvm::dyn_cast<PointerType>(QT.getTypePtr())) {
    // Normalize `X*` to `const void*`, invoke `printValue(const void**)`,
    // unless it's a character string.
    QualType QTPointeeUnqual = PT->getPointeeType().getUnqualifiedType();
    if (!Ctx.hasSameType(QTPointeeUnqual, Ctx.CharTy) &&
        !Ctx.hasSameType(QTPointeeUnqual, Ctx.WCharTy) &&
        !Ctx.hasSameType(QTPointeeUnqual, Ctx.Char16Ty) &&
        !Ctx.hasSameType(QTPointeeUnqual, Ctx.Char32Ty)) {
      QT = Ctx.getPointerType(Ctx.VoidTy.withConst());
    }
  } else if (const auto *RTy = llvm::dyn_cast<ReferenceType>(QT.getTypePtr())) {
    // X& will be printed as X* (the pointer will be added below).
    QT = RTy->getPointeeType();
    // Val will be a X**, but we cast this to X*, so dereference here:
    ValPtr = *(const void *const *)ValPtr;
  }

  // `PrintValueRuntime()` takes the *address* of the value to be printed:
  QualType QTPtr = Ctx.getPointerType(QT);
  Expr *TypeArg = CStyleCastPtrExpr(S, QTPtr, (uintptr_t)ValPtr);
  llvm::SmallVector<Expr *, 1> CallArgs = {TypeArg};

  // Create the CallExpr.
  ExprResult RuntimeCall =
      S.ActOnCallExpr(S.getCurScope(), OverldExpr, SourceLocation(), CallArgs,
                      SourceLocation());
  assert(!RuntimeCall.isInvalid() && "Cannot create call to PrintValueRuntime");

  // Create the ReturnStmt.
  StmtResult RetStmt =
      S.ActOnReturnStmt(SourceLocation(), RuntimeCall.get(), S.getCurScope());
  assert(!RetStmt.isInvalid() && "Cannot create ReturnStmt");

  // Create the CompoundStmt.
  StmtResult Body =
      CompoundStmt::Create(Ctx, {RetStmt.get()}, FPOptionsOverride(),
                           SourceLocation(), SourceLocation());
  assert(!Body.isInvalid() && "Cannot create function body");

  WrapperFD->setBody(Body.get());
  // Add attribute `__attribute__((used))`.
  WrapperFD->addAttr(UsedAttr::CreateImplicit(Ctx));
}

static constexpr const char *const WrapperName = "__InterpreterCallPrint";

static llvm::Expected<llvm::orc::ExecutorAddr> CompileDecl(Interpreter &Interp,
                                                           Decl *D) {
  assert(D && "The Decl being compiled can't be null");

  ASTConsumer &Consumer = Interp.getCompilerInstance()->getASTConsumer();
  Consumer.HandleTopLevelDecl(DeclGroupRef(D));
  Interp.getCompilerInstance()->getSema().PerformPendingInstantiations();
  ASTContext &C = Interp.getASTContext();
  TranslationUnitDecl *TUPart = C.getTranslationUnitDecl();
  assert(!TUPart->containsDecl(D) && "Decl already added!");
  TUPart->addDecl(D);
  Consumer.HandleTranslationUnit(C);

  if (std::unique_ptr<llvm::Module> M = Interp.GenModule()) {
    PartialTranslationUnit PTU = {TUPart, std::move(M)};
    if (llvm::Error Err = Interp.Execute(PTU))
      return Err;
    ASTNameGenerator ASTNameGen(Interp.getASTContext());
    llvm::Expected<llvm::orc::ExecutorAddr> AddrOrErr =
        Interp.getSymbolAddressFromLinkerName(ASTNameGen.getName(D));

    return AddrOrErr;
  }
  return llvm::orc::ExecutorAddr{};
}

static std::string CreateUniqName(std::string Base) {
  static size_t I = 0;
  Base += std::to_string(I);
  I += 1;
  return Base;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const void *Ptr) {
  return PrintAddress(Ptr, '@');
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const void **Ptr) {
  return PrintAddress(*Ptr, '@');
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const bool *Val) {
  if (*Val)
    return "true";
  return "false";
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const char *Val) {
  return PrintOneChar(*Val);
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const signed char *Val) {
  return PrintOneChar(*Val);
}

REPL_EXTERNAL_VISIBILITY std::string
PrintValueRuntime(const unsigned char *Val) {
  return PrintOneChar(*Val);
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const short *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY
std::string PrintValueRuntime(const unsigned short *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const int *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY
std::string PrintValueRuntime(const unsigned int *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const long *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const long long *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY
std::string PrintValueRuntime(const unsigned long *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY
std::string PrintValueRuntime(const unsigned long long *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << *Val;
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const float *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << llvm::format("%#.6g", *Val) << 'f';
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const double *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << llvm::format("%#.12g", *Val);
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const long double *Val) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << llvm::format("%#.8Lg", *Val) << 'L';
  return Str;
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const char *const *Val) {
  return PrintString(Val);
}

REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const char **Val) {
  return PrintString(Val);
}

namespace clang {
std::string Interpreter::ValueDataToString(const Value &V) {
  QualType QT = V.getType();
  QualType DesugaredTy = QT.getDesugaredType(V.getASTContext());
  QualType NonRefTy = DesugaredTy.getNonReferenceType();

  if (NonRefTy->isEnumeralType())
    return PrintEnum(V);

  if (NonRefTy->isFunctionType())
    return PrintFunction(V, &V);

  if ((NonRefTy->isPointerType() || NonRefTy->isMemberPointerType()) &&
      NonRefTy->getPointeeType()->isFunctionProtoType())
    return PrintFunction(V, V.getPtr());

  if (NonRefTy->isNullPtrType())
    return "nullptr\n";

  // If it is a builtin type dispatch to the builtin overloads.
  if (auto *BT = DesugaredTy.getCanonicalType()->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
    default:
      return "{ unknown builtin type: }" + std::to_string(BT->getKind());
#define X(type, name)                                                          \
  case clang::BuiltinType::name: {                                             \
    type val = V.get##name();                                                  \
    return PrintValueRuntime(&val);                                            \
  }
      REPL_BUILTIN_TYPES
#undef X
    }
  }
  if (auto *CXXRD = NonRefTy->getAsCXXRecordDecl())
    if (CXXRD->isLambda())
      return PrintAddress(V.getPtr(), '@');

  // All fails then generate a runtime call, this is slow.
  Sema &S = getCompilerInstance()->getSema();
  ASTContext &Ctx = S.getASTContext();

  QualType RetTy;
  if (Ctx.getLangOpts().CPlusPlus && !StdString) {

    // Only include the header on demand because it's very heavy.
    if (llvm::Error E = ParseAndExecute(
            "#include <__clang_interpreter_runtime_printvalue.h>")) {
      llvm::logAllUnhandledErrors(std::move(E), llvm::errs(), "Parsing failed");
      return "{Internal error}";
    }

    // Find and cache std::string.
    NamespaceDecl *Std = LookupNamespace(S, "std");
    assert(Std && "Cannot find namespace std");
    Decl *StdStringDecl = LookupNamed(S, "string", Std);
    assert(StdStringDecl && "Cannot find std::string");
    const auto *StdStringTyDecl = llvm::dyn_cast<TypeDecl>(StdStringDecl);
    assert(StdStringTyDecl && "Cannot find type of std::string");
    RetTy = QualType(StdStringTyDecl->getTypeForDecl(), /*Quals=*/0);
  } else {
    RetTy = Ctx.getPointerType(Ctx.CharTy.withConst());
  }

  // Create the wrapper function.
  DeclarationName DeclName = &Ctx.Idents.get(CreateUniqName(WrapperName));
  QualType FnTy =
      Ctx.getFunctionType(RetTy, {}, FunctionProtoType::ExtProtoInfo());
  auto *WrapperFD = FunctionDecl::Create(
      Ctx, Ctx.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
      DeclName, FnTy, Ctx.getTrivialTypeSourceInfo(FnTy), SC_None);

  void *ValPtr = V.getPtr();

  // FIXME: We still need to understand why we have to get the pointer to the
  // underlying Value storage for this to work reliabily...
  if (!V.isManuallyAlloc())
    ValPtr = V.getPtrAddress();

  BuildWrapperBody(*this, S, Ctx, WrapperFD, V.getType(), ValPtr);

  auto AddrOrErr = CompileDecl(*this, WrapperFD);
  if (!AddrOrErr)
    llvm::logAllUnhandledErrors(AddrOrErr.takeError(), llvm::errs(),
                                "Fail to get symbol address");
  if (auto *Main = AddrOrErr->toPtr<std::string (*)()>())
    return (*Main)();
  return "Unable to print the value!";
}

std::string Interpreter::ValueTypeToString(const Value &V) const {
  ASTContext &Ctx = const_cast<ASTContext &>(V.getASTContext());
  QualType QT = V.getType();

  std::string QTStr = PrintQualType(Ctx, QT);

  if (QT->isReferenceType())
    QTStr += " &";

  return QTStr;
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::CompileDtorCall(CXXRecordDecl *CXXRD) {
  assert(CXXRD && "Cannot compile a destructor for a nullptr");
  if (auto Dtor = Dtors.find(CXXRD); Dtor != Dtors.end())
    return Dtor->getSecond();

  if (CXXRD->hasIrrelevantDestructor())
    return llvm::orc::ExecutorAddr{};

  CXXDestructorDecl *DtorRD =
      getCompilerInstance()->getSema().LookupDestructor(CXXRD);

  llvm::StringRef Name =
      getCodeGen()->GetMangledName(GlobalDecl(DtorRD, Dtor_Base));
  auto AddrOrErr = getSymbolAddress(Name);
  if (!AddrOrErr)
    return AddrOrErr.takeError();

  Dtors[CXXRD] = *AddrOrErr;
  return AddrOrErr;
}

enum InterfaceKind { NoAlloc, WithAlloc, CopyArray, NewTag };

class InterfaceKindVisitor
    : public TypeVisitor<InterfaceKindVisitor, InterfaceKind> {

  Sema &S;
  Expr *E;
  llvm::SmallVectorImpl<Expr *> &Args;

public:
  InterfaceKindVisitor(Sema &S, Expr *E, llvm::SmallVectorImpl<Expr *> &Args)
      : S(S), E(E), Args(Args) {}

  InterfaceKind computeInterfaceKind(QualType Ty) {
    return Visit(Ty.getTypePtr());
  }

  InterfaceKind VisitRecordType(const RecordType *Ty) {
    return InterfaceKind::WithAlloc;
  }

  InterfaceKind VisitMemberPointerType(const MemberPointerType *Ty) {
    return InterfaceKind::WithAlloc;
  }

  InterfaceKind VisitConstantArrayType(const ConstantArrayType *Ty) {
    return InterfaceKind::CopyArray;
  }

  InterfaceKind VisitFunctionProtoType(const FunctionProtoType *Ty) {
    HandlePtrType(Ty);
    return InterfaceKind::NoAlloc;
  }

  InterfaceKind VisitPointerType(const PointerType *Ty) {
    HandlePtrType(Ty);
    return InterfaceKind::NoAlloc;
  }

  InterfaceKind VisitReferenceType(const ReferenceType *Ty) {
    ExprResult AddrOfE = S.CreateBuiltinUnaryOp(SourceLocation(), UO_AddrOf, E);
    assert(!AddrOfE.isInvalid() && "Can not create unary expression");
    Args.push_back(AddrOfE.get());
    return InterfaceKind::NoAlloc;
  }

  InterfaceKind VisitBuiltinType(const BuiltinType *Ty) {
    if (Ty->isNullPtrType())
      Args.push_back(E);
    else if (Ty->isFloatingType())
      Args.push_back(E);
    else if (Ty->isIntegralOrEnumerationType())
      HandleIntegralOrEnumType(Ty);
    else if (Ty->isVoidType()) {
      // Do we need to still run `E`?
    }

    return InterfaceKind::NoAlloc;
  }

  InterfaceKind VisitEnumType(const EnumType *Ty) {
    HandleIntegralOrEnumType(Ty);
    return InterfaceKind::NoAlloc;
  }

private:
  // Force cast these types to the uint that fits the register size. That way we
  // reduce the number of overloads of `__clang_Interpreter_SetValueNoAlloc`.
  void HandleIntegralOrEnumType(const Type *Ty) {
    ASTContext &Ctx = S.getASTContext();
    uint64_t PtrBits = Ctx.getTypeSize(Ctx.VoidPtrTy);
    QualType UIntTy = Ctx.getBitIntType(/*Unsigned=*/true, PtrBits);
    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(UIntTy);
    ExprResult CastedExpr =
        S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E);
    assert(!CastedExpr.isInvalid() && "Cannot create cstyle cast expr");
    Args.push_back(CastedExpr.get());
  }

  void HandlePtrType(const Type *Ty) {
    ASTContext &Ctx = S.getASTContext();
    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ctx.VoidPtrTy);
    ExprResult CastedExpr =
        S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E);
    assert(!CastedExpr.isInvalid() && "Can not create cstyle cast expression");
    Args.push_back(CastedExpr.get());
  }
};

// This synthesizes a call expression to a speciall
// function that is responsible for generating the Value.
// In general, we transform:
//   clang-repl> x
// To:
//   // 1. If x is a built-in type like int, float.
//   __clang_Interpreter_SetValueNoAlloc(ThisInterp, OpaqueValue, xQualType, x);
//   // 2. If x is a struct, and a lvalue.
//   __clang_Interpreter_SetValueNoAlloc(ThisInterp, OpaqueValue, xQualType,
//   &x);
//   // 3. If x is a struct, but a rvalue.
//   new (__clang_Interpreter_SetValueWithAlloc(ThisInterp, OpaqueValue,
//   xQualType)) (x);
llvm::Expected<Expr *> Interpreter::AttachValuePrinting(Expr *E) {
  Sema &S = getCompilerInstance()->getSema();
  ASTContext &Ctx = S.getASTContext();

  // Find the value printing builtins.
  if (!ValuePrintingInfo[0]) {
    assert(llvm::all_of(ValuePrintingInfo, [](Expr *E) { return !E; }));

    auto LookupInterface = [&](Expr *&Interface,
                               llvm::StringRef Name) -> llvm::Error {
      LookupResult R(S, &Ctx.Idents.get(Name), SourceLocation(),
                     Sema::LookupOrdinaryName,
                     RedeclarationKind::ForVisibleRedeclaration);
      S.LookupQualifiedName(R, Ctx.getTranslationUnitDecl());
      if (R.empty())
        return llvm::make_error<llvm::StringError>(
            Name + " not found!", llvm::inconvertibleErrorCode());

      CXXScopeSpec CSS;
      Interface = S.BuildDeclarationNameExpr(CSS, R, /*ADL=*/false).get();
      return llvm::Error::success();
    };
    static constexpr llvm::StringRef Builtin[] = {
        "__clang_Interpreter_SetValueNoAlloc",
        "__clang_Interpreter_SetValueWithAlloc",
        "__clang_Interpreter_SetValueCopyArr", "__ci_newtag"};
    if (llvm::Error Err =
            LookupInterface(ValuePrintingInfo[NoAlloc], Builtin[NoAlloc]))
      return std::move(Err);

    if (Ctx.getLangOpts().CPlusPlus) {
      if (llvm::Error Err =
              LookupInterface(ValuePrintingInfo[WithAlloc], Builtin[WithAlloc]))
        return std::move(Err);
      if (llvm::Error Err =
              LookupInterface(ValuePrintingInfo[CopyArray], Builtin[CopyArray]))
        return std::move(Err);
      if (llvm::Error Err =
              LookupInterface(ValuePrintingInfo[NewTag], Builtin[NewTag]))
        return std::move(Err);
    }
  }

  llvm::SmallVector<Expr *, 4> AdjustedArgs;
  // Create parameter `ThisInterp`.
  AdjustedArgs.push_back(CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)this));

  // Create parameter `OutVal`.
  AdjustedArgs.push_back(
      CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)&LastValue));

  // Build `__clang_Interpreter_SetValue*` call.

  // Get rid of ExprWithCleanups.
  if (auto *EWC = llvm::dyn_cast_if_present<ExprWithCleanups>(E))
    E = EWC->getSubExpr();

  QualType Ty = E->getType();
  QualType DesugaredTy = Ty.getDesugaredType(Ctx);

  // For lvalue struct, we treat it as a reference.
  if (DesugaredTy->isRecordType() && E->isLValue()) {
    DesugaredTy = Ctx.getLValueReferenceType(DesugaredTy);
    Ty = Ctx.getLValueReferenceType(Ty);
  }

  Expr *TypeArg =
      CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)Ty.getAsOpaquePtr());
  // The QualType parameter `OpaqueType`, represented as `void*`.
  AdjustedArgs.push_back(TypeArg);

  // We push the last parameter based on the type of the Expr. Note we need
  // special care for rvalue struct.
  InterfaceKindVisitor V(S, E, AdjustedArgs);
  Scope *Scope = nullptr;
  ExprResult SetValueE;
  InterfaceKind Kind = V.computeInterfaceKind(DesugaredTy);
  switch (Kind) {
  case InterfaceKind::WithAlloc:
    LLVM_FALLTHROUGH;
  case InterfaceKind::CopyArray: {
    // __clang_Interpreter_SetValueWithAlloc.
    ExprResult AllocCall =
        S.ActOnCallExpr(Scope, ValuePrintingInfo[InterfaceKind::WithAlloc],
                        E->getBeginLoc(), AdjustedArgs, E->getEndLoc());
    assert(!AllocCall.isInvalid() && "Can't create runtime interface call!");

    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ty, SourceLocation());

    // Force CodeGen to emit destructor.
    if (auto *RD = Ty->getAsCXXRecordDecl()) {
      auto *Dtor = S.LookupDestructor(RD);
      Dtor->addAttr(UsedAttr::CreateImplicit(Ctx));
      getCompilerInstance()->getASTConsumer().HandleTopLevelDecl(
          DeclGroupRef(Dtor));
    }

    // __clang_Interpreter_SetValueCopyArr.
    if (Kind == InterfaceKind::CopyArray) {
      const auto *ConstantArrTy =
          cast<ConstantArrayType>(DesugaredTy.getTypePtr());
      size_t ArrSize = Ctx.getConstantArrayElementCount(ConstantArrTy);
      Expr *ArrSizeExpr = IntegerLiteralExpr(Ctx, ArrSize);
      Expr *Args[] = {E, AllocCall.get(), ArrSizeExpr};
      SetValueE =
          S.ActOnCallExpr(Scope, ValuePrintingInfo[InterfaceKind::CopyArray],
                          SourceLocation(), Args, SourceLocation());
    }
    Expr *Args[] = {AllocCall.get(), ValuePrintingInfo[InterfaceKind::NewTag]};
    ExprResult CXXNewCall = S.BuildCXXNew(
        E->getSourceRange(),
        /*UseGlobal=*/true, /*PlacementLParen=*/SourceLocation(), Args,
        /*PlacementRParen=*/SourceLocation(),
        /*TypeIdParens=*/SourceRange(), TSI->getType(), TSI, std::nullopt,
        E->getSourceRange(), E);

    assert(!CXXNewCall.isInvalid() &&
           "Can't create runtime placement new call!");

    SetValueE = S.ActOnFinishFullExpr(CXXNewCall.get(),
                                      /*DiscardedValue=*/false);
    break;
  }
  // __clang_Interpreter_SetValueNoAlloc.
  case InterfaceKind::NoAlloc: {
    SetValueE =
        S.ActOnCallExpr(Scope, ValuePrintingInfo[InterfaceKind::NoAlloc],
                        E->getBeginLoc(), AdjustedArgs, E->getEndLoc());
    break;
  }
  default:
    llvm_unreachable("Unhandled InterfaceKind");
  }

  // It could fail, like printing an array type in C. (not supported)
  if (SetValueE.isInvalid())
    return E;

  return SetValueE.get();
}

} // namespace clang

// Temporary rvalue struct that need special care.
REPL_EXTERNAL_VISIBILITY void *
__clang_Interpreter_SetValueWithAlloc(void *This, void *OutVal,
                                      void *OpaqueType) {
  Value &VRef = *(Value *)OutVal;
  VRef = Value(static_cast<Interpreter *>(This), OpaqueType);
  return VRef.getPtr();
}

extern "C" void REPL_EXTERNAL_VISIBILITY __clang_Interpreter_SetValueNoAlloc(
    void *This, void *OutVal, void *OpaqueType, ...) {
  Value &VRef = *(Value *)OutVal;
  Interpreter *I = static_cast<Interpreter *>(This);
  VRef = Value(I, OpaqueType);
  if (VRef.isVoid())
    return;

  va_list args;
  va_start(args, /*last named param*/ OpaqueType);

  QualType QT = VRef.getType();
  if (VRef.getKind() == Value::K_PtrOrObj) {
    VRef.setPtr(va_arg(args, void *));
  } else {
    if (const auto *ET = QT->getAs<EnumType>())
      QT = ET->getDecl()->getIntegerType();
    switch (QT->castAs<BuiltinType>()->getKind()) {
    default:
      llvm_unreachable("unknown type kind!");
      break;
      // Types shorter than int are resolved as int, else va_arg has UB.
    case BuiltinType::Bool:
      VRef.setBool(va_arg(args, int));
      break;
    case BuiltinType::Char_S:
      VRef.setChar_S(va_arg(args, int));
      break;
    case BuiltinType::SChar:
      VRef.setSChar(va_arg(args, int));
      break;
    case BuiltinType::Char_U:
      VRef.setChar_U(va_arg(args, unsigned));
      break;
    case BuiltinType::UChar:
      VRef.setUChar(va_arg(args, unsigned));
      break;
    case BuiltinType::Short:
      VRef.setShort(va_arg(args, int));
      break;
    case BuiltinType::UShort:
      VRef.setUShort(va_arg(args, unsigned));
      break;
    case BuiltinType::Int:
      VRef.setInt(va_arg(args, int));
      break;
    case BuiltinType::UInt:
      VRef.setUInt(va_arg(args, unsigned));
      break;
    case BuiltinType::Long:
      VRef.setLong(va_arg(args, long));
      break;
    case BuiltinType::ULong:
      VRef.setULong(va_arg(args, unsigned long));
      break;
    case BuiltinType::LongLong:
      VRef.setLongLong(va_arg(args, long long));
      break;
    case BuiltinType::ULongLong:
      VRef.setULongLong(va_arg(args, unsigned long long));
      break;
      // Types shorter than double are resolved as double, else va_arg has UB.
    case BuiltinType::Float:
      VRef.setFloat(va_arg(args, double));
      break;
    case BuiltinType::Double:
      VRef.setDouble(va_arg(args, double));
      break;
    case BuiltinType::LongDouble:
      VRef.setLongDouble(va_arg(args, long double));
      break;
      // See REPL_BUILTIN_TYPES.
    }
  }
  va_end(args);
}

// A trampoline to work around the fact that operator placement new cannot
// really be forward declared due to libc++ and libstdc++ declaration mismatch.
// FIXME: __clang_Interpreter_NewTag is ODR violation because we get the same
// definition in the interpreter runtime. We should move it in a runtime header
// which gets included by the interpreter and here.
struct __clang_Interpreter_NewTag {};
REPL_EXTERNAL_VISIBILITY void *
operator new(size_t __sz, void *__p, __clang_Interpreter_NewTag) noexcept {
  // Just forward to the standard operator placement new.
  return operator new(__sz, __p);
}
