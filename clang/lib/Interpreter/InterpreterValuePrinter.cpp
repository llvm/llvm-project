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

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
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

static std::string EnumToString(ASTContext &Ctx, QualType QT, uint64_t Data) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);

  bool IsFirst = true;
  llvm::APSInt AP = Ctx.MakeIntValue(Data, QT);

  auto *ED = QT->castAsEnumDecl();
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

static std::string FunctionToString(ASTContext &Ctx, QualType QT,
                                    const void *Ptr) {
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << "Function @" << Ptr;

  const DeclContext *PTU = Ctx.getTranslationUnitDecl();
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
    const auto *Arg = InterfaceCall->getArg(InterfaceCall->getNumArgs() - 1);

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

static std::string CharPtrToString(const char *Ptr) {
  if (!Ptr)
    return "0";

  std::string Result = "\"";
  Result += Ptr;
  Result += '"';
  return Result;
}

namespace clang {

std::string ValueToString::toString(const Value *Buf) {

  switch (Buf->getKind()) {
  case Value::K_Builtin:
    return BuiltinToString(*Buf);
  case Value::K_Pointer:
    return PointerToString(*Buf);
  case Value::K_Str:
    break;
  case Value::K_Array:
    return ArrayToString(*Buf);

  default:
    break;
  }
  return "";
}

std::string ValueToString::BuiltinToString(const Value &B) {
  if (!B.hasValue())
    return ""; // No data in buffer

  QualType QT = B.getType();
  QualType DesugaredTy = QT.getDesugaredType(Ctx);
  QualType NonRefTy = DesugaredTy.getNonReferenceType();

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
      SS << ((B.getBool()) ? "true" : "false");
      return Str;
    case clang::BuiltinType::Char_S:
      SS << '\'' << B.getChar_S() << '\'';
      return Str;
    case clang::BuiltinType::SChar:
      SS << '\'' << B.getSChar() << '\'';
      return Str;
    case clang::BuiltinType::Char_U:
      SS << '\'' << B.getChar_U() << '\'';
      return Str;
    case clang::BuiltinType::UChar:
      SS << '\'' << B.getUChar() << '\'';
      return Str;
    case clang::BuiltinType::Short:
      SS << B.getShort();
      return Str;
    case clang::BuiltinType::UShort:
      SS << B.getUShort();
      return Str;
    case clang::BuiltinType::Int:
      SS << B.getInt();
      return Str;
    case clang::BuiltinType::UInt:
      SS << B.getUInt();
      return Str;
    case clang::BuiltinType::Long:
      SS << B.getLong();
      return Str;
    case clang::BuiltinType::ULong:
      SS << B.getULong();
      return Str;
    case clang::BuiltinType::LongLong:
      SS << B.getLongLong();
      return Str;
    case clang::BuiltinType::ULongLong:
      SS << B.getULongLong();
      return Str;
    case clang::BuiltinType::Float:
      return formatFloating(B.getFloat(), /*suffix=*/'f');

    case clang::BuiltinType::Double:
      return formatFloating(B.getDouble());

    case clang::BuiltinType::LongDouble:
      return formatFloating(B.getLongDouble(), /*suffix=*/'L');
    }
  }

  if (NonRefTy->isEnumeralType())
    return EnumToString(Ctx, QT, B.getUInt());

  return "";
}

std::string ValueToString::PointerToString(const Value &P) {

  QualType QT = P.getType();
  QualType DesugaredTy = QT.getDesugaredType(Ctx);
  QualType NonRefTy = DesugaredTy.getNonReferenceType();

  if ((NonRefTy->isPointerType() || NonRefTy->isMemberPointerType()) &&
      NonRefTy->getPointeeType()->isFunctionProtoType())
    return FunctionToString(Ctx, QT, (void *)P.getAddr());

  if (NonRefTy->isFunctionType())
    return FunctionToString(Ctx, QT, (void *)P.getAddr());

  if (NonRefTy->isNullPtrType())
    return "nullptr\n";

  if (NonRefTy->isPointerType()) {
    auto PointeeTy = NonRefTy->getPointeeType();

    // char* -> print string literal
    if (PointeeTy->isCharType()) {
      if (P.HasPointee() && P.getPointerPointee().isStr())
        return CharPtrToString(P.getPointerPointee().getStrVal());
      return std::to_string(P.getAddr());
    }

    std::ostringstream OS;
    OS << "0x" << std::hex << P.getAddr();
    return OS.str();
  }

  std::ostringstream OS;
  OS << "@0x" << std::hex << P.getAddr();
  return OS.str();
}

std::string ValueToString::ArrayToString(const Value &A) {
  if (const ConstantArrayType *CAT = Ctx.getAsConstantArrayType(A.getType())) {
    QualType ElemTy = CAT->getElementType();
    // Treat null terminated char arrays as strings basically.
    if (ElemTy->isCharType()) {
      const Value &LastV =
          A.getArrayInitializedElt(A.getArrayInitializedElts() - 1);
      if (LastV.hasValue() && !LastV.isAbsent() && LastV.isBuiltin()) {
        char last = LastV.getChar_S();
        if (last != '\0')
          goto not_a_string;
      }
      std::string Res;
      Res += "\"";
      for (size_t I = 0, N = A.getArraySize(); I < N; ++I) {
        const Value &EleVal = A.getArrayInitializedElt(I);
        if (EleVal.hasValue() && EleVal.isBuiltin()) {
          char c = EleVal.getChar_S();
          if (c != '\0')
            Res += c;
        }
      }
      Res += "\"";
      return Res;
    }
  }
not_a_string:
  std::ostringstream OS;

  OS << "{ ";
  for (size_t I = 0, N = A.getArraySize(); I < N; ++I) {
    const Value &EleVal = A.getArrayInitializedElt(I);
    if (EleVal.hasValue())
      OS << this->toString(&EleVal);

    if (I + 1 < N)
      OS << ", ";
  }

  OS << " }";
  return OS.str();
}

std::string ValueToString::toString(QualType QT) {
  ASTContext &AstCtx = const_cast<ASTContext &>(Ctx);

  std::string QTStr = QualTypeToString(AstCtx, QT);

  if (QT->isReferenceType())
    QTStr += " &";

  return QTStr;
}

void ValueResultManager::resetAndDump() {
  if (!LastVal.hasValue() || LastVal.isAbsent())
    return;

  QualType Ty = LastVal.getType();

  Value V = std::move(LastVal);

  // Don't even try to print a void or an invalid type, it doesn't make sense.
  if (Ty->isVoidType())
    return;

  // We need to get all the results together then print it, since `printType` is
  // much faster than `printData`.
  std::string Str;
  llvm::raw_string_ostream SS(Str);
  ValueToString ValToStr(Ctx);
  SS << "(";
  SS << ValToStr.toString(Ty);
  SS << ") ";
  SS << ValToStr.toString(&V);
  SS << "\n";
  llvm::outs() << Str;
  V.clear();
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
  Sema &S;
  ASTContext &Ctx;

public:
  ExprConverter(Sema &S, ASTContext &Ctx) : S(S), Ctx(Ctx) {}

private:
  bool isAddressOfExpr(Expr *E) {
    if (!E)
      return false;
    if (auto *UO = dyn_cast<UnaryOperator>(E))
      return UO->getOpcode() == UO_AddrOf;
    if (auto *ICE = dyn_cast<ImplicitCastExpr>(E))
      return isAddressOfExpr(ICE->getSubExpr());
    return false;
  }

  /// Build a single &E using Sema.
  ExprResult buildAddrOfWithSema(Expr *E,
                                 SourceLocation Loc = SourceLocation()) {
    // Sema will materialize temporaries as necessary.
    ExprResult Res = S.CreateBuiltinUnaryOp(Loc, UO_AddrOf, E);
    if (Res.isInvalid())
      return ExprError();
    return Res;
  }

public:
  /// Create (&E) as a void* (uses Sema for & creation)
  ExprResult CreateAddressOfVoidPtrExpr(QualType Ty, Expr *ForCast,
                                        bool takeAddr = false) {
    ExprResult AddrOfRes = ForCast;

    if (takeAddr) {
      // don't create & twice
      if (!isAddressOfExpr(ForCast)) {
        AddrOfRes = buildAddrOfWithSema(ForCast);
        if (AddrOfRes.isInvalid())
          return ExprError();
      } else {
        // already an &-expression; keep it as-is
        AddrOfRes = ForCast;
      }
    }

    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(Ctx.VoidPtrTy);
    ExprResult CastedExpr = S.BuildCStyleCastExpr(
        SourceLocation(), TSI, SourceLocation(), AddrOfRes.get());
    if (CastedExpr.isInvalid())
      return ExprError();

    return CastedExpr;
  }

  /// Wrap rvalues in a temporary (var) so they become addressable.
  Expr *CreateMaterializeTemporaryExpr(Expr *E) {
    return S.CreateMaterializeTemporaryExpr(E->getType(), E,
                                            /*BoundToLvalueReference=*/true);
  }

  ExprResult makeScalarAddressable(QualType Ty, Expr *E) {
    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(Ty, E, /*takeAddr=*/true);
    return CreateAddressOfVoidPtrExpr(Ty, CreateMaterializeTemporaryExpr(E),
                                      /*takeAddr=*/true);
  }

  ExprResult handleBuiltinTypeExpr(const BuiltinType *, QualType QTy, Expr *E) {
    return makeScalarAddressable(QTy, E);
  }

  ExprResult handleEnumTypeExpr(const EnumType *, QualType QTy, Expr *E) {
    uint64_t PtrBits = Ctx.getTypeSize(Ctx.VoidPtrTy);
    QualType UIntTy = Ctx.getBitIntType(/*Unsigned=*/true, PtrBits);
    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(UIntTy);
    ExprResult CastedExpr =
        S.BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), E);
    assert(!CastedExpr.isInvalid() && "Cannot create cstyle cast expr");
    return makeScalarAddressable(QTy, CastedExpr.get());
  }

  ExprResult handlePointerTypeExpr(const PointerType *, QualType QTy, Expr *E) {
    // Pointer expressions always evaluate to a pointer value.
    // No need to take address or materialize.
    return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/false);
  }

  ExprResult handleArrayTypeExpr(const ConstantArrayType *, QualType QTy,
                                 Expr *E) {
    if (isa<StringLiteral>(E)) {
      if (Ctx.getLangOpts().CPlusPlus)
        return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/true);
      return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/false);
    }

    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/true);
    return CreateAddressOfVoidPtrExpr(QTy, E,
                                      /*takeAddr=*/false);
  }

  ExprResult handleFunctionTypeExpr(const FunctionType *, QualType QTy,
                                    Expr *E) {
    return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/false);
  }

  ExprResult handleMemberPointerTypeExpr(const Type *, QualType QTy, Expr *E) {
    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/true);
    return CreateAddressOfVoidPtrExpr(QTy, CreateMaterializeTemporaryExpr(E),
                                      /*takeAddr=*/true);
  }

  ExprResult handleRecordTypeExpr(const Type *, QualType QTy, Expr *E) {
    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/true);

    TypeSourceInfo *TSI = Ctx.getTrivialTypeSourceInfo(QTy, SourceLocation());
    ExprResult CXXNewCall =
        S.BuildCXXNew(E->getSourceRange(),
                      /*UseGlobal=*/true, /*PlacementLParen=*/SourceLocation(),
                      MultiExprArg(),
                      /*PlacementRParen=*/SourceLocation(),
                      /*TypeIdParens=*/SourceRange(), TSI->getType(), TSI,
                      std::nullopt, E->getSourceRange(), E);

    if (CXXNewCall.isInvalid())
      return ExprError();

    auto CallRes = S.ActOnFinishFullExpr(CXXNewCall.get(),
                                         /*DiscardedValue=*/false);
    if (CallRes.isInvalid())
      return ExprError();
    return CreateAddressOfVoidPtrExpr(QTy, CallRes.get(), /*takeAddr=*/false);
  }

  ExprResult handleAnyObjectExpr(const Type *, QualType QTy, Expr *E) {
    if (E->isLValue() || E->isXValue())
      return CreateAddressOfVoidPtrExpr(QTy, E, /*takeAddr=*/true);
    auto Res = S.CreateBuiltinUnaryOp(SourceLocation(), UO_AddrOf, E);
    assert(!Res.isInvalid());
    return CreateAddressOfVoidPtrExpr(QTy, Res.get(),
                                      /*takeAddr=*/false);
  }
};

class InterfaceKindVisitor : public TypeVisitor<InterfaceKindVisitor, bool> {
  Sema &S;
  Expr *E;
  llvm::SmallVectorImpl<Expr *> &Args;
  ExprConverter Converter;

public:
  InterfaceKindVisitor(Sema &S, Expr *E, llvm::SmallVectorImpl<Expr *> &Args)
      : S(S), E(E), Args(Args), Converter(S, S.getASTContext()) {}

  bool transformExpr(QualType Ty) { return Visit(Ty.getTypePtr()); }

  bool VisitType(const Type *T) {
    Args.push_back(Converter.handleAnyObjectExpr(T, QualType(T, 0), E).get());
    return true;
  }

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

  bool VisitMemberPointerType(const MemberPointerType *Ty) {
    Args.push_back(
        Converter.handleMemberPointerTypeExpr(Ty, QualType(Ty, 0), E).get());
    return true;
  }

  bool VisitRecordType(const RecordType *Ty) {
    Args.push_back(
        Converter.handleRecordTypeExpr(Ty, QualType(Ty, 0), E).get());
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

  bool VisitFunctionType(const FunctionType *Ty) {
    Args.push_back(
        Converter.handleFunctionTypeExpr(Ty, QualType(Ty, 0), E).get());
    return true;
  }

  bool VisitEnumType(const EnumType *Ty) {
    Args.push_back(Converter.handleEnumTypeExpr(Ty, QualType(Ty, 0), E).get());
    return true;
  }
};

enum RunTimeFnTag {
  OrcSendResult,
  ClangSendResult,
  ClangDestroyObj,
  OrcRunDtorWrapper,
  ClangRunDtorWrapper
};

static constexpr llvm::StringRef RunTimeFnTagName[] = {
    "__orc_rt_SendResultValue", "__clang_Interpreter_SendResultValue",
    "__clang_Interpreter_destroyObj", "__orc_rt_runDtor",
    "__clang_Interpreter_runDtor"};

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

  std::optional<ValueCleanup> CleanUp = std::nullopt;
  if (DesugaredTy->isRecordType() && E->isPRValue())
    if (auto *CXXRD = DesugaredTy->getAsCXXRecordDecl()) {
      auto *Dtor = S.LookupDestructor(CXXRD);
      Dtor->addAttr(UsedAttr::CreateImplicit(Ctx));
      getCompilerInstance()->getASTConsumer().HandleTopLevelDecl(
          DeclGroupRef(Dtor));

      auto ObjDtor =
          [this](QualType Ty) -> llvm::Expected<llvm::orc::ExecutorAddr> {
        if (auto *CXXRD = Ty->getAsCXXRecordDecl())
          return this->CompileDtorCall(CXXRD);
        return llvm::make_error<llvm::StringError>(
            "destructor not found!", llvm::inconvertibleErrorCode());
      };

      std::string DestroyObjFnName = RunTimeFnTagName[ClangDestroyObj].str();
      std::string RunDtorWrapperFnName =
          RunTimeFnTagName[isOOP ? OrcRunDtorWrapper : ClangRunDtorWrapper]
              .str();

      // #if defined(__APPLE__)
      //       // On macOS, runtime symbols may require a leading underscore
      //       DestroyObjFnName.insert(0, "_");
      //       RunDtorWrapperFnName.insert(0, "_");
      // #endif

      auto RunDtorWrapperAddr = getSymbolAddress(RunDtorWrapperFnName);
      if (!RunDtorWrapperAddr)
        return RunDtorWrapperAddr.takeError();

      auto DestroyObjAddr = getSymbolAddress(DestroyObjFnName);
      if (!DestroyObjAddr)
        return DestroyObjAddr.takeError();

      if (auto E = getExecutionEngine()) {
        CleanUp = std::make_optional<ValueCleanup>(
            &E->getExecutionSession(), *RunDtorWrapperAddr, *DestroyObjAddr,
            std::move(ObjDtor));
      }
    }

  auto ID = ValMgr->registerPendingResult(Ty, std::move(CleanUp));

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

REPL_EXTERNAL_VISIBILITY llvm::orc::shared::CWrapperFunctionResult
__clang_Interpreter_runDtor(char *ArgData, size_t ArgSize) {
  return llvm::orc::shared::WrapperFunction<llvm::orc::shared::SPSError(
      llvm::orc::shared::SPSExecutorAddr, llvm::orc::shared::SPSExecutorAddr)>::
      handle(ArgData, ArgSize,
             [](llvm::orc::ExecutorAddr DtorFn,
                llvm::orc::ExecutorAddr This) -> llvm::Error {
               DtorFn.toPtr<void (*)(unsigned char *)>()(
                   This.toPtr<unsigned char *>());
               return llvm::Error::success();
             })
          .release();
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
