//===---- MatchSwitch.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the `MatchSwitch` abstraction for building a "switch"
//  statement, where each case of the switch is defined by an AST matcher. The
//  cases are considered in order, like pattern matching in functional
//  languages.
//
//  Currently, the design is catered towards simplifying the implementation of
//  `DataflowAnalysis` transfer functions. Based on experience here, this
//  library may be generalized and moved to ASTMatchers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_MATCHSWITCH_H_
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_MATCHSWITCH_H_

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {

/// A common form of state shared between the cases of a transfer function.
template <typename LatticeT> struct TransferState {
  TransferState(LatticeT &Lattice, Environment &Env)
      : Lattice(Lattice), Env(Env) {}

  /// Current lattice element.
  LatticeT &Lattice;
  Environment &Env;
};

/// Matches against `Stmt` and, based on its structure, dispatches to an
/// appropriate handler.
template <typename State>
using MatchSwitch = std::function<void(const Stmt &, ASTContext &, State &)>;

/// Collects cases of a "match switch": a collection of matchers paired with
/// callbacks, which together define a switch that can be applied to a
/// `Stmt`. This structure can simplify the definition of `transfer` functions
/// that rely on pattern-matching.
///
/// For example, consider an analysis that handles particular function calls. It
/// can define the `MatchSwitch` once, in the constructor of the analysis, and
/// then reuse it each time that `transfer` is called, with a fresh state value.
///
/// \code
/// MatchSwitch<TransferState<MyLattice> BuildSwitch() {
///   return MatchSwitchBuilder<TransferState<MyLattice>>()
///     .CaseOf(callExpr(callee(functionDecl(hasName("foo")))), TransferFooCall)
///     .CaseOf(callExpr(argumentCountIs(2),
///                      callee(functionDecl(hasName("bar")))),
///             TransferBarCall)
///     .Build();
/// }
/// \endcode
template <typename State> class MatchSwitchBuilder {
public:
  // An action is triggered by the match of a pattern against the input
  // statement. For generality, actions take both the matched statement and the
  // set of bindings produced by the match.
  using Action = std::function<void(
      const Stmt *, const ast_matchers::MatchFinder::MatchResult &, State &)>;

  MatchSwitchBuilder &&CaseOf(ast_matchers::internal::Matcher<Stmt> M,
                              Action A) && {
    Matchers.push_back(std::move(M));
    Actions.push_back(std::move(A));
    return std::move(*this);
  }

  // Convenience function for the common case, where bound nodes are not
  // needed. `Node` should be a subclass of `Stmt`.
  template <typename Node>
  MatchSwitchBuilder &&CaseOf(ast_matchers::internal::Matcher<Stmt> M,
                              void (*Action)(const Node *, State &)) && {
    Matchers.push_back(std::move(M));
    Actions.push_back([Action](const Stmt *Stmt,
                               const ast_matchers::MatchFinder::MatchResult &,
                               State &S) { Action(cast<Node>(Stmt), S); });
    return std::move(*this);
  }

  MatchSwitch<State> Build() && {
    return [Matcher = BuildMatcher(), Actions = std::move(Actions)](
               const Stmt &Stmt, ASTContext &Context, State &S) {
      auto Results = ast_matchers::matchDynamic(Matcher, Stmt, Context);
      if (Results.empty())
        return;
      // Look through the map for the first binding of the form "TagN..." use
      // that to select the action.
      for (const auto &Element : Results[0].getMap()) {
        llvm::StringRef ID(Element.first);
        size_t Index = 0;
        if (ID.consume_front("Tag") && !ID.getAsInteger(10, Index) &&
            Index < Actions.size()) {
          Actions[Index](
              &Stmt,
              ast_matchers::MatchFinder::MatchResult(Results[0], &Context), S);
          return;
        }
      }
    };
  }

private:
  ast_matchers::internal::DynTypedMatcher BuildMatcher() {
    using ast_matchers::anything;
    using ast_matchers::stmt;
    using ast_matchers::unless;
    using ast_matchers::internal::DynTypedMatcher;
    if (Matchers.empty())
      return stmt(unless(anything()));
    for (int I = 0, N = Matchers.size(); I < N; ++I) {
      std::string Tag = ("Tag" + llvm::Twine(I)).str();
      // Many matchers are not bindable, so ensure that tryBind will work.
      Matchers[I].setAllowBind(true);
      auto M = *Matchers[I].tryBind(Tag);
      // Each anyOf explicitly controls the traversal kind. The anyOf itself is
      // set to `TK_AsIs` to ensure no nodes are skipped, thereby deferring to
      // the kind of the branches. Then, each branch is either left as is, if
      // the kind is already set, or explicitly set to `TK_AsIs`. We choose this
      // setting because it is the default interpretation of matchers.
      Matchers[I] =
          !M.getTraversalKind() ? M.withTraversalKind(TK_AsIs) : std::move(M);
    }
    // The matcher type on the cases ensures that `Expr` kind is compatible with
    // all of the matchers.
    return DynTypedMatcher::constructVariadic(
        DynTypedMatcher::VO_AnyOf, ASTNodeKind::getFromNodeKind<Stmt>(),
        std::move(Matchers));
  }

  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
  std::vector<Action> Actions;
};
} // namespace dataflow
} // namespace clang
#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_MATCHSWITCH_H_
