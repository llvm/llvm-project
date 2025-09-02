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

#include "IncrementalAction.h"
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
#include <cmath>
#include <cstdarg>
#include <sstream>
#include <string>

#define DEBUG_TYPE "interp-value"

using namespace clang;

static std::string DeclTypeToString(const QualType &QT, NamedDecl *D) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  if (QT.hasQualifiers())
    SS << QT.getQualifiers().getAsString() << " ";
  SS << D->getQualifiedNameAsString();
  return Str;
}

static std::string QualTypeToString(ASTContext &Ctx, QualType QT) {
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
    return DeclTypeToString(NonRefTy, TTy->getOriginalDecl());

  if (const auto *TRy = dyn_cast<RecordType>(NonRefTy))
    return DeclTypeToString(NonRefTy, TRy->getOriginalDecl());

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
    return DeclTypeToString(NonRefTy, TDTy->getDecl());
  }
  return GetFullTypeName(Ctx, NonRefTy);
}

static std::string EnumToString(const Value &V) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  ASTContext &Ctx = const_cast<ASTContext &>(V.getASTContext());

  uint64_t Data = V.convertTo<uint64_t>();
  bool IsFirst = true;
  llvm::APSInt AP = Ctx.MakeIntValue(Data, V.getType());

  auto *ED = V.getType()->castAsEnumDecl();
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
  SS << " : " << QualTypeToString(Ctx, ED->getIntegerType()) << " " << APStr;
  return Str;
}

static std::string FunctionToString(const Value &V, const void *Ptr) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "Function @" << Ptr;

  const DeclContext *PTU = V.getASTContext().getTranslationUnitDecl();
  // Find the last top-level-stmt-decl. This is a forward iterator but the
  // partial translation unit should not be large.
  const TopLevelStmtDecl *TLSD = nullptr;
  for (const Decl *D : PTU->noload_decls())
    if (isa<TopLevelStmtDecl>(D))
      TLSD = cast<TopLevelStmtDecl>(D);

  // Get __clang_Interpreter_SetValueNoAlloc(void *This, void *OutVal, void
  // *OpaqueType, void *Val);
  const FunctionDecl *FD = nullptr;
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

static std::string VoidPtrToString(const void *Ptr) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << Ptr;
  return Str;
}

static std::string CharPtrToString(const char *Ptr) {
  if (!Ptr)
    return "0";

  std::string Result = "\"";
  Result += Ptr;
  Result += '"';
  return Result;
}

namespace clang {

struct ValueRef : public Value {
  ValueRef(const Interpreter *In, void *Ty) : Value(In, Ty) {
    // Tell the base class to not try to deallocate if it manages the value.
    IsManuallyAlloc = false;
  }
};

std::string Interpreter::ValueDataToString(const Value &V) const {
  Sema &S = getCompilerInstance()->getSema();
  ASTContext &Ctx = S.getASTContext();

  QualType QT = V.getType();

  if (const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(QT)) {
    QualType ElemTy = CAT->getElementType();
    size_t ElemCount = Ctx.getConstantArrayElementCount(CAT);
    const Type *BaseTy = CAT->getBaseElementTypeUnsafe();
    size_t ElemSize = Ctx.getTypeSizeInChars(BaseTy).getQuantity();

    // Treat null terminated char arrays as strings basically.
    if (ElemTy->isCharType()) {
      char last = *(char *)(((uintptr_t)V.getPtr()) + ElemCount * ElemSize - 1);
      if (last == '\0')
        return CharPtrToString((char *)V.getPtr());
    }

    std::string Result = "{ ";
    for (unsigned Idx = 0, N = CAT->getZExtSize(); Idx < N; ++Idx) {
      ValueRef InnerV = ValueRef(this, ElemTy.getAsOpaquePtr());
      if (ElemTy->isBuiltinType()) {
        // Single dim arrays, advancing.
        uintptr_t Offset = (uintptr_t)V.getPtr() + Idx * ElemSize;
        InnerV.setRawBits((void *)Offset, ElemSize * 8);
      } else {
        // Multi dim arrays, position to the next dimension.
        size_t Stride = ElemCount / N;
        uintptr_t Offset = ((uintptr_t)V.getPtr()) + Idx * Stride * ElemSize;
        InnerV.setPtr((void *)Offset);
      }

      Result += ValueDataToString(InnerV);

      // Skip the \0 if the char types
      if (Idx < N - 1)
        Result += ", ";
    }
    Result += " }";
    return Result;
  }

  QualType DesugaredTy = QT.getDesugaredType(Ctx);
  QualType NonRefTy = DesugaredTy.getNonReferenceType();

  // FIXME: Add support for user defined printers.
  // LookupResult R = LookupUserDefined(S, QT);
  // if (!R.empty())
  //   return CallUserSpecifiedPrinter(R, V);

  // If it is a builtin type dispatch to the builtin overloads.
  if (auto *BT = DesugaredTy.getCanonicalType()->getAs<BuiltinType>()) {

    auto formatFloating = [](auto Val, char Suffix = '\0') -> std::string {
      std::string Out;
      llvm::raw_string_ostream SS(Out);

      if (std::isnan(Val) || std::isinf(Val)) {
        SS << llvm::format("%g", Val);
        return SS.str();
      }
      if (Val == static_cast<decltype(Val)>(static_cast<int64_t>(Val)))
        SS << llvm::format("%.1f", Val);
      else if (std::abs(Val) < 1e-4 || std::abs(Val) > 1e6 || Suffix == 'f')
        SS << llvm::format("%#.6g", Val);
      else if (Suffix == 'L')
        SS << llvm::format("%#.12Lg", Val);
      else
        SS << llvm::format("%#.8g", Val);

      if (Suffix != '\0')
        SS << Suffix;
      return SS.str();
    };

    std::string Str;
    llvm::raw_string_ostream SS(Str);
    switch (BT->getKind()) {
    default:
      return "{ error: unknown builtin type '" + std::to_string(BT->getKind()) +
             " '}";
    case clang::BuiltinType::Bool:
      SS << ((V.getBool()) ? "true" : "false");
      return Str;
    case clang::BuiltinType::Char_S:
      SS << '\'' << V.getChar_S() << '\'';
      return Str;
    case clang::BuiltinType::SChar:
      SS << '\'' << V.getSChar() << '\'';
      return Str;
    case clang::BuiltinType::Char_U:
      SS << '\'' << V.getChar_U() << '\'';
      return Str;
    case clang::BuiltinType::UChar:
      SS << '\'' << V.getUChar() << '\'';
      return Str;
    case clang::BuiltinType::Short:
      SS << V.getShort();
      return Str;
    case clang::BuiltinType::UShort:
      SS << V.getUShort();
      return Str;
    case clang::BuiltinType::Int:
      SS << V.getInt();
      return Str;
    case clang::BuiltinType::UInt:
      SS << V.getUInt();
      return Str;
    case clang::BuiltinType::Long:
      SS << V.getLong();
      return Str;
    case clang::BuiltinType::ULong:
      SS << V.getULong();
      return Str;
    case clang::BuiltinType::LongLong:
      SS << V.getLongLong();
      return Str;
    case clang::BuiltinType::ULongLong:
      SS << V.getULongLong();
      return Str;
    case clang::BuiltinType::Float:
      return formatFloating(V.getFloat(), /*suffix=*/'f');

    case clang::BuiltinType::Double:
      return formatFloating(V.getDouble());

    case clang::BuiltinType::LongDouble:
      return formatFloating(V.getLongDouble(), /*suffix=*/'L');
    }
  }

  if ((NonRefTy->isPointerType() || NonRefTy->isMemberPointerType()) &&
      NonRefTy->getPointeeType()->isFunctionProtoType())
    return FunctionToString(V, V.getPtr());

  if (NonRefTy->isFunctionType())
    return FunctionToString(V, &V);

  if (NonRefTy->isEnumeralType())
    return EnumToString(V);

  if (NonRefTy->isNullPtrType())
    return "nullptr\n";

  // FIXME: Add support for custom printers in C.
  if (NonRefTy->isPointerType()) {
    if (NonRefTy->getPointeeType()->isCharType())
      return CharPtrToString((char *)V.getPtr());

    return VoidPtrToString(V.getPtr());
  }

  // Fall back to printing just the address of the unknown object.
  return "@" + VoidPtrToString(V.getPtr());
}

std::string ValueResultManager::ValueTypeToString(QualType QT) const {
  ASTContext &AstCtx = const_cast<ASTContext &>(Ctx);

  std::string QTStr = QualTypeToString(AstCtx, QT);

  if (QT->isReferenceType())
    QTStr += " &";

  return QTStr;
}

void ValueResultManager::resetAndDump() {
  if (!ValBuf)
    return;

  QualType Ty = ValBuf->Ty;

  std::unique_ptr<ValueBuffer> Val = nullptr;
  ValBuf.swap(Val);

  // Don't even try to print a void or an invalid type, it doesn't make sense.
  if (Ty->isVoidType() || !Val->isValid())
    return;

  // We need to get all the results together then print it, since `printType` is
  // much faster than `printData`.
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  SS << "(";
  SS << ValueTypeToString(Ty);
  SS << ") ";
  SS << Val->toString();
  SS << "\n";
  llvm::outs() << Str;
}

llvm::Expected<llvm::orc::ExecutorAddr>
Interpreter::CompileDtorCall(CXXRecordDecl *CXXRD) const {
  assert(CXXRD && "Cannot compile a destructor for a nullptr");
  if (auto Dtor = Dtors.find(CXXRD); Dtor != Dtors.end())
    return Dtor->getSecond();

  if (CXXRD->hasIrrelevantDestructor())
    return llvm::orc::ExecutorAddr{};

  CXXDestructorDecl *DtorRD =
      getCompilerInstance()->getSema().LookupDestructor(CXXRD);

  llvm::StringRef Name =
      Act->getCodeGen()->GetMangledName(GlobalDecl(DtorRD, Dtor_Base));
  auto AddrOrErr = getSymbolAddress(Name);
  if (!AddrOrErr)
    return AddrOrErr.takeError();

  Dtors[CXXRD] = *AddrOrErr;
  return AddrOrErr;
}

class ExprConverter {
  ASTContext &Ctx;

public:
  ExprConverter(ASTContext &Ctx) : Ctx(Ctx) {}

  /// Create (&E) as a void*
  ExprResult CreateAddressOfVoidPtrExpr(QualType Ty, Expr *ForCast) {
    QualType VoidPtrTy = Ctx.getPointerType(Ctx.VoidTy);

    // &E
    Expr *AddrOf = UnaryOperator::Create(
        Ctx, ForCast, UO_AddrOf, Ctx.getPointerType(Ty), VK_PRValue,
        OK_Ordinary, SourceLocation(), false, FPOptionsOverride());

    // static_cast<void*>(&E)
    return CXXStaticCastExpr::Create(
        Ctx, VoidPtrTy, VK_PRValue, CK_BitCast, AddrOf, nullptr,
        Ctx.getTrivialTypeSourceInfo(VoidPtrTy), FPOptionsOverride(),
        SourceLocation(), SourceLocation(), SourceRange());
  }

  /// Create a temporary VarDecl with initializer.
  VarDecl *createTempVarDecl(QualType Ty, llvm::StringRef BaseName,
                             Expr *Init) {
    static unsigned Counter = 0;
    IdentifierInfo &Id = Ctx.Idents.get((BaseName + Twine(++Counter)).str());

    VarDecl *VD = VarDecl::Create(Ctx, Ctx.getTranslationUnitDecl(),
                                  SourceLocation(), SourceLocation(), &Id, Ty,
                                  Ctx.getTrivialTypeSourceInfo(Ty), SC_Auto);

    VD->setInit(Init);
    VD->setInitStyle(VarDecl::CInit);
    VD->markUsed(Ctx);

    return VD;
  }

  /// Wrap rvalues in a temporary so they become addressable.
  Expr *CreateMaterializeTemporaryExpr(Expr *E) {
    return new (Ctx) MaterializeTemporaryExpr(E->getType(), E,
                                              /*BoundToLvalueReference=*/true);
  }

  /// Generic helper: materialize if needed, then &expr as void*.
  ExprResult makeAddressable(QualType QTy, Expr *E) {
    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(QTy, E);

    if (E->isPRValue())
      return CreateAddressOfVoidPtrExpr(QTy, CreateMaterializeTemporaryExpr(E));

    return ExprError();
  }

  ExprResult handleBuiltinTypeExpr(const BuiltinType *, QualType QTy, Expr *E) {
    return makeAddressable(QTy, E);
  }
  ExprResult handlePointerTypeExpr(const PointerType *, QualType QTy, Expr *E) {
    return makeAddressable(QTy, E);
  }
  ExprResult handleArrayTypeExpr(const ConstantArrayType *, QualType QTy,
                                 Expr *E) {
    return makeAddressable(QTy, E);
  }
};

class InterfaceKindVisitor : public TypeVisitor<InterfaceKindVisitor, bool> {
  Sema &S;
  Expr *E;
  llvm::SmallVectorImpl<Expr *> &Args;
  ExprConverter Converter;

public:
  InterfaceKindVisitor(Sema &S, Expr *E, llvm::SmallVectorImpl<Expr *> &Args)
      : S(S), E(E), Args(Args), Converter(S.getASTContext()) {}

  bool transformExpr(QualType Ty) { return Visit(Ty.getTypePtr()); }

  bool VisitBuiltinType(const BuiltinType *Ty) {
    if (Ty->isNullPtrType()) {
      Args.push_back(E);
    } else if (Ty->isFloatingType() || Ty->isIntegralOrEnumerationType()) {
      Args.push_back(
          Converter.handleBuiltinTypeExpr(Ty, QualType(Ty, 0), E).get());
    } else if (Ty->isVoidType()) {
      return false;
    }
    return true;
  }

  bool VisitPointerType(const PointerType *Ty) {
    Args.push_back(
        Converter.handlePointerTypeExpr(Ty, QualType(Ty, 0), E).get());
    return true;
  }

  bool VisitConstantArrayType(const ConstantArrayType *Ty) {
    Args.push_back(Converter.handleArrayTypeExpr(Ty, QualType(Ty, 0), E).get());
    return true;
  }

  bool VisitReferenceType(const ReferenceType *Ty) {
    ExprResult AddrOfE = S.CreateBuiltinUnaryOp(SourceLocation(), UO_AddrOf, E);
    assert(!AddrOfE.isInvalid() && "Cannot create unary expression");
    Args.push_back(AddrOfE.get());
    return true;
  }

  bool VisitRecordType(const RecordType *) { return true; }
  bool VisitMemberPointerType(const MemberPointerType *) { return true; }
  bool VisitFunctionType(const FunctionType *) { return true; }
  bool VisitEnumType(const EnumType *) { return true; }
};

enum RunTimeFnTag { OrcSendResult, ClangSendResult };

static constexpr llvm::StringRef RunTimeFnTagName[] = {
    "__orc_rt_SendResultValue", "__clang_Interpreter_SendResultValue"};

// This synthesizes a call expression to a speciall
// function that is responsible for generating the Value.
// In general, we transform c++:
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
llvm::Expected<Expr *> Interpreter::convertExprToValue(Expr *E, bool isOOP) {
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

    if (llvm::Error Err = LookupInterface(ValuePrintingInfo[OrcSendResult],
                                          RunTimeFnTagName[OrcSendResult]))
      return std::move(Err);

    if (llvm::Error Err = LookupInterface(ValuePrintingInfo[ClangSendResult],
                                          RunTimeFnTagName[ClangSendResult]))
      return std::move(Err);
  }

  llvm::SmallVector<Expr *, 4> AdjustedArgs;

  if (!isOOP)
    // Create parameter `ValMgr`.
    AdjustedArgs.push_back(
        CStyleCastPtrExpr(S, Ctx.VoidPtrTy, (uintptr_t)ValMgr.get()));

  // Get rid of ExprWithCleanups.
  if (auto *EWC = llvm::dyn_cast_if_present<ExprWithCleanups>(E))
    E = EWC->getSubExpr();

  QualType Ty = E->IgnoreImpCasts()->getType();
  QualType DesugaredTy = Ty.getDesugaredType(Ctx);

  // For lvalue struct, we treat it as a reference.
  if (DesugaredTy->isRecordType() && E->isLValue()) {
    DesugaredTy = Ctx.getLValueReferenceType(DesugaredTy);
    Ty = Ctx.getLValueReferenceType(Ty);
  }

  auto ID = ValMgr->registerPendingResult(Ty);

  AdjustedArgs.push_back(IntegerLiteralExpr(Ctx, ID));

  // We push the last parameter based on the type of the Expr. Note we need
  // special care for rvalue struct.
  InterfaceKindVisitor V(S, E, AdjustedArgs);
  ExprResult SetValueE;
  Scope *Scope = nullptr;
  if (!V.transformExpr(DesugaredTy))
    return E;

  RunTimeFnTag Tag =
      isOOP ? RunTimeFnTag::OrcSendResult : RunTimeFnTag::ClangSendResult;
  SetValueE = S.ActOnCallExpr(Scope, ValuePrintingInfo[Tag], E->getBeginLoc(),
                              AdjustedArgs, E->getEndLoc());

  // It could fail, like printing an array type in C. (not supported)
  if (SetValueE.isInvalid())
    return E;

  return SetValueE.get();
}
} // namespace clang

using namespace clang;

// Temporary rvalue struct that need special care.
extern "C" {
REPL_EXTERNAL_VISIBILITY void
__clang_Interpreter_SendResultValue(void *Ctx, uint64_t Id, void *Addr) {
  static_cast<ValueResultManager *>(Ctx)->deliverResult(
      [](llvm::Error Err) { llvm::cantFail(std::move(Err)); }, Id,
      llvm::orc::ExecutorAddr::fromPtr(Addr));
}
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
