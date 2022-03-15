//===-- CIRGenFunction.h - Per-Function state for CIR gen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for CIR translation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
#define LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H

#include "CIRGenCall.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"

#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/TargetInfo.h"

namespace clang {
class Expr;
} // namespace clang

namespace mlir {
namespace func {
class CallOp;
}
} // namespace mlir

namespace cir {

// FIXME: for now we are reusing this from lib/Clang/CodeGenFunction.h, which
// isn't available in the include dir. Same for getEvaluationKind below.
enum TypeEvaluationKind { TEK_Scalar, TEK_Complex, TEK_Aggregate };

class CIRGenFunction {
public:
  enum class EvaluationOrder {
    ///! No langauge constraints on evaluation order.
    Default,
    ///! Language semantics requrie left-to-right evaluation
    ForceLeftToRight,
    ///! Language semantics require right-to-left evaluation.
    ForceRightToLeft
  };

  clang::QualType FnRetQualTy;
  std::optional<mlir::Type> FnRetTy;
  std::optional<mlir::Value> FnRetAlloca;

  CIRGenModule &CGM;

  // CurFuncDecl - Holds the Decl for the current outermost non-closure context
  const clang::Decl *CurFuncDecl;

  // The CallExpr within the current statement that the musttail attribute
  // applies to. nullptr if there is no 'musttail' on the current statement.
  const clang::CallExpr *MustTailCall = nullptr;

  clang::ASTContext &getContext() const;

  /// Sanitizers enabled for this function.
  clang::SanitizerSet SanOpts;

  ///  Return the TypeEvaluationKind of QualType \c T.
  static TypeEvaluationKind getEvaluationKind(clang::QualType T);

  static bool hasScalarEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Scalar;
  }

  static bool hasAggregateEvaluationKind(clang::QualType T) {
    return getEvaluationKind(T) == TEK_Aggregate;
  }

  CIRGenFunction(CIRGenModule &CGM);

  CIRGenTypes &getTypes() const { return CGM.getTypes(); }

  const clang::LangOptions &getLangOpts() const { return CGM.getLangOpts(); }

  // TODO: This is currently just a dumb stub. But we want to be able to clearly
  // assert where we arne't doing things that we know we should and will crash
  // as soon as we add a DebugInfo type to this class.
  std::nullptr_t *getDebugInfo() { return nullptr; }

  // Wrapper for function prototype sources. Wraps either a FunctionProtoType or
  // an ObjCMethodDecl.
  struct PrototypeWrapper {
    llvm::PointerUnion<const clang::FunctionProtoType *,
                       const clang::ObjCMethodDecl *>
        P;

    PrototypeWrapper(const clang::FunctionProtoType *FT) : P(FT) {}
    PrototypeWrapper(const clang::ObjCMethodDecl *MD) : P(MD) {}
  };

  /// An abstract representation of regular/ObjC call/message targets.
  class AbstractCallee {
    /// The function declaration of the callee.
    const clang::Decl *CalleeDecl;

  public:
    AbstractCallee() : CalleeDecl(nullptr) {}
    AbstractCallee(const clang::FunctionDecl *FD) : CalleeDecl(FD) {}
    AbstractCallee(const clang::ObjCMethodDecl *OMD) : CalleeDecl(OMD) {}
    bool hasFunctionDecl() const {
      return llvm::isa_and_nonnull<clang::FunctionDecl>(CalleeDecl);
    }
    const clang::Decl *getDecl() const { return CalleeDecl; }
    unsigned getNumParams() const {
      if (const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(CalleeDecl))
        return FD->getNumParams();
      return llvm::cast<clang::ObjCMethodDecl>(CalleeDecl)->param_size();
    }
    const clang::ParmVarDecl *getParamDecl(unsigned I) const {
      if (const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(CalleeDecl))
        return FD->getParamDecl(I);
      return *(llvm::cast<clang::ObjCMethodDecl>(CalleeDecl)->param_begin() +
               I);
    }
  };

  void buildCallArgs(
      CallArgList &Args, PrototypeWrapper Prototype,
      llvm::iterator_range<clang::CallExpr::const_arg_iterator> ArgRange,
      AbstractCallee AC = AbstractCallee(), unsigned ParamsToSkip = 0,
      EvaluationOrder Order = EvaluationOrder::Default);

  /// buildCall - Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  RValue buildCall(const CIRGenFunctionInfo &CallInfo,
                   const CIRGenCallee &Callee, ReturnValueSlot ReturnValue,
                   const CallArgList &Args, mlir::func::CallOp &callOrInvoke,
                   bool IsMustTail, clang::SourceLocation Loc);
  RValue buildCall(clang::QualType FnType, const CIRGenCallee &Callee,
                   const clang::CallExpr *E, ReturnValueSlot returnValue,
                   mlir::Value Chain = nullptr);

  RValue buildCallExpr(const clang::CallExpr *E,
                       ReturnValueSlot ReturnValue = ReturnValueSlot());

  void buildCallArg(CallArgList &args, const clang::Expr *E,
                    clang::QualType ArgType);

  /// buildAnyExprToTemp - Similarly to buildAnyExpr(), however, the result will
  /// always be accessible even if no aggregate location is provided.
  RValue buildAnyExprToTemp(const clang::Expr *E);

  CIRGenCallee buildCallee(const clang::Expr *E);

  /// buildAnyExpr - Emit code to compute the specified expression which can
  /// have any type. The result is returned as an RValue struct. If this is an
  /// aggregate expression, the aggloc/agglocvolatile arguments indicate where
  /// the result should be returned.
  RValue buildAnyExpr(const clang::Expr *E,
                      AggValueSlot aggSlot = AggValueSlot::ignored(),
                      bool ignoreResult = false);

  mlir::Type convertType(clang::QualType T);
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_CIRGENFUNCTION_H
