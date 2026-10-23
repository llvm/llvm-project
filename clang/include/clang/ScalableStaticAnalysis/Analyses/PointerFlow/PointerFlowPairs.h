//===- PointerFlowPairs.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file provides PointerFlowPair and PointerFlowPairMatcher.
//
// PointerFlowPair represents an element '(l, r)' of the pointer-flow relation
// over declarations and expressions of pointer/array type. Each pair
// corresponds to a value-copying (or "assignment"-flavored) language construct
// (e.g. an assignment, argument passing, a return, or an initialization).  It
// requires that if 'l's type is refined to carry a property (e.g., buffer
// bounds), then 'r's type must follow; otherwise the property would be
// lost in value-copy from 'r' to 'l'.
//
// PointerFlowPairMatcher walks an AST node and collects the PointerFlowPairs it
// generates. It outputs matched pairs '(l, r)' such that
// - 'l' and 'r' have compatible types;
// - 'l' is either a pointer or an array;
// - 'r' may be a list-initializer, when it has an array type.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_POINTERFLOW_POINTERFLOWPAIRS_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_POINTERFLOW_POINTERFLOWPAIRS_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/SmallVector.h"
#include <type_traits>

namespace clang::ssaf {

/// Data structure representing a pointer flow.
/// Invariant: LHS and RHS should have compatible types.
struct PointerFlowPair {
  /// The left-hand side of an assignment or a variable/field
  /// definition, a formal parameter, or the function owning a return
  /// stmt:
  llvm::PointerUnion<const ValueDecl *, const Expr *> LHS;
  /// The right-hand side of an assignment or a variable/field
  /// definition, an actual argument, or the expr being returning:
  const Expr *RHS;
  /// True iff the left-hand side of this PointerFlowPair represents the
  /// return entity of a callable:
  bool IsLHSRet;

  PointerFlowPair(const ValueDecl *LHS, const Expr *RHS, bool IsLHSRet = false)
      : LHS(LHS), RHS(RHS), IsLHSRet(IsLHSRet) {
    assert((!IsLHSRet || isa_and_nonnull<FunctionDecl>(LHS)) &&
           "IsLHSRet -> LHS is a FunctionDecl");
  }

  PointerFlowPair(const Expr *LHS, const Expr *RHS)
      : LHS(LHS), RHS(RHS), IsLHSRet(false) {}

  /// An alternative to access the PointerUnion LHS directly---handle it using
  /// a function object that defines:
  /// - T operator()(const ValueDecl *, bool IsRet, Args...);
  /// - T operator()(const Expr *, Args...);
  template <
      typename F, typename... Args,
      typename T = std::invoke_result_t<F, const ValueDecl *, bool, Args...>>
  T visitLHS(F &&Visitor, Args... ExtraArgs) const {
    static_assert(std::is_invocable_r_v<T, F, const ValueDecl *, bool, Args...>,
                  "Visitor(const ValueDecl *, bool, Args...) must return T");
    static_assert(std::is_invocable_r_v<T, F, const Expr *, Args...>,
                  "Visitor(const Expr *, Args...) must return T");
    if (const auto *VD = LHS.dyn_cast<const ValueDecl *>())
      return Visitor(VD, IsLHSRet, ExtraArgs...);
    return Visitor(LHS.dyn_cast<const Expr *>(), ExtraArgs...);
  }
};

class PointerFlowPairMatcher {
public:
  ASTContext &Ctx;
  PointerFlowPairMatcher(ASTContext &Ctx) : Ctx(Ctx) {}

  // FIXME: Known gaps -- the following constructs are not handled:
  //   - Lambda captures (by-copy, by-reference, or init-capture) of a
  //   pointer.
  //   - Structured bindings (`auto [a, b] = pair;`) -- the per-element
  //     `BindingDecl`s are neither `VarDecl` nor `FieldDecl`.

  /// Match and collect pointer flow.
  /// The macth function 'F' can be described by the following rules:
  ///
  /// F(l = r)          := (l, r), if 'l' has a pointer/array type;
  ///                   := F(field_1, list_item_1), ..., if 'l' has a record
  ///                                                    type and 'r' is a
  ///                                                    list-initializer
  /// F(foo(a, b, ...)) := F(Param_1 = a), F(Param_2 = b), ...
  /// F(return e;)      := F(FunRet = e), where 'FunRet' is the return
  ///                                          entity of the enclosing
  ///                                          function
  /// F(ctor(a, ...) : x1(y1), ... {...})
  ///                   := F(Param_1 = a), ...,
  ///                            F(x1 = y1), ....
  /// F(T var = e)      := F(var = e)
  ///
  /// \param DynNode the node being matched.
  /// \param Contributor the Decl that contributes \c DynNode; it is the
  /// enclosing function decl if \c DynNode is a return stmt.
  /// \param Result output, a set of \c PointerFlowPair matched from \c
  /// DynNode
  ///
  /// Upon return, each pair '(l, r)' in \c Result is must have the following
  /// properties:
  /// - 'l' and 'r' have compatible types;
  /// - 'l' is either a pointer or an array;
  /// - 'r' may be a list-initializer when 'l' is an array
  bool matches(const DynTypedNode &DynNode, const NamedDecl *Contributor,
               llvm::SmallVectorImpl<PointerFlowPair> &Result) const;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_POINTERFLOW_POINTERFLOWPAIRS_H
