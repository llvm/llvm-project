//=== AnyCall.h - Abstraction over different callables --------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility class for performing generic operations over different callables.
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_CLANG_ANALYSIS_ANY_CALL_H
#define LLVM_CLANG_ANALYSIS_ANY_CALL_H

#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"

namespace clang {

/// An instance of this class corresponds to a 'callable' call.
class AnyCall {
public:
  enum Kind {
    /// A function, function pointer, or a C++ method call
    Function,

    /// A call to an Objective-C method
    ObjCMethod,

    /// A call to an Objective-C block
    Block,

    /// An implicit C++ destructor call (called implicitly
    /// or by operator 'delete')
    Destructor,

    /// An implicit or explicit C++ constructor call
    Constructor,

    /// A C++ allocation function call (operator `new`), via C++ new-expression
    Allocator,

    /// A C++ deallocation function call (operator `delete`), via C++
    /// delete-expression
    Deallocator
  };

private:
  /// Call expression, remains null iff the call is an implicit destructor call.
  const Expr *E = nullptr;

  /// Corresponds to a statically known declaration of the called function,
  /// or null if it is not known (e.g. for a function pointer).
  const Decl *D = nullptr;
  Kind K;

  AnyCall(const Expr *E, const Decl *D, Kind K) : E(E), D(D), K(K) {}

public:
  AnyCall(const CallExpr *CE) : E(CE) {
    D = CE->getCalleeDecl();
    K = (CE->getCallee()->getType()->getAs<BlockPointerType>()) ? Block
                                                                : Function;
    if (D && ((K == Function && !isa<FunctionDecl>(D)) ||
              (K == Block && !isa<BlockDecl>(D))))
      D = nullptr;
  }

  AnyCall(const ObjCMessageExpr *ME)
      : E(ME), D(ME->getMethodDecl()), K(ObjCMethod) {}

  AnyCall(const CXXNewExpr *NE)
      : E(NE), D(NE->getOperatorNew()), K(Allocator) {}

  AnyCall(const CXXDeleteExpr *NE)
      : E(NE), D(NE->getOperatorDelete()), K(Deallocator) {}

  AnyCall(const CXXConstructExpr *NE)
      : E(NE), D(NE->getConstructor()), K(Constructor) {}

  /// If {@code E} is a generic call (to ObjC method /function/block/etc),
  /// return a constructed {@code AnyCall} object. Return None otherwise.
  static Optional<AnyCall> forExpr(const Expr *E) {
    if (const auto *ME = dyn_cast<ObjCMessageExpr>(E)) {
      return AnyCall(ME);
    } else if (const auto *CE = dyn_cast<CallExpr>(E)) {
      return AnyCall(CE);
    } else if (const auto *CXNE = dyn_cast<CXXNewExpr>(E)) {
      return AnyCall(CXNE);
    } else if (const auto *CXDE = dyn_cast<CXXDeleteExpr>(E)) {
      return AnyCall(CXDE);
    } else if (const auto *CXCE = dyn_cast<CXXConstructExpr>(E)) {
      return AnyCall(CXCE);
    } else {
      return None;
    }
  }

  static AnyCall forDestructorCall(const CXXDestructorDecl *D) {
    return AnyCall(/*E=*/nullptr, D, Destructor);
  }

  /// \returns formal parameters for direct calls (including virtual calls)
  ArrayRef<ParmVarDecl *> parameters() const {
    if (!D)
      return None;

    if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
      return FD->parameters();
    } else if (const auto *MD = dyn_cast<ObjCMethodDecl>(D)) {
      return MD->parameters();
    } else if (const auto *CD = dyn_cast<CXXConstructorDecl>(D)) {
      return CD->parameters();
    } else if (const auto *BD = dyn_cast<BlockDecl>(D)) {
      return BD->parameters();
    } else {
      return None;
    }
  }

  using param_const_iterator = ArrayRef<ParmVarDecl *>::const_iterator;
  param_const_iterator param_begin() const { return parameters().begin(); }
  param_const_iterator param_end() const { return parameters().end(); }
  size_t param_size() const { return parameters().size(); }
  bool param_empty() const { return parameters().empty(); }

  QualType getReturnType(ASTContext &Ctx) const {
    switch (K) {
    case Function:
    case Block:
      return cast<CallExpr>(E)->getCallReturnType(Ctx);
    case ObjCMethod:
      return cast<ObjCMessageExpr>(E)->getCallReturnType(Ctx);
    case Destructor:
    case Constructor:
    case Allocator:
    case Deallocator:
      return cast<FunctionDecl>(D)->getReturnType();
    }
  }

  /// \returns Function identifier if it is a named declaration,
  /// {@code nullptr} otherwise.
  const IdentifierInfo *getIdentifier() const {
    if (const auto *ND = dyn_cast_or_null<NamedDecl>(D))
      return ND->getIdentifier();
    return nullptr;
  }

  const Decl *getDecl() const {
    return D;
  }

  const Expr *getExpr() const {
    return E;
  }

  Kind getKind() const {
    return K;
  }

  void dump() const {
    if (E)
      E->dump();
    if (D)
      D->dump();
  }
};

}

#endif // LLVM_CLANG_ANALYSIS_ANY_CALL_H
