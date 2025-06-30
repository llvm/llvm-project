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
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

#include <cstdarg>

namespace clang {

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
llvm::Expected<Expr *> Interpreter::ExtractValueFromExpr(Expr *E) {
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

using namespace clang;

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
