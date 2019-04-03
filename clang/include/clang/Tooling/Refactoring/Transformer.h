//===--- Transformer.h - Clang source-rewriting library ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  Defines a library supporting the concise specification of clang-based
///  source-to-source transformations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_TRANSFORMER_H_
#define LLVM_CLANG_TOOLING_REFACTOR_TRANSFORMER_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include <deque>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace clang {
namespace tooling {
/// Determines the part of the AST node to replace.  We support this to work
/// around the fact that the AST does not differentiate various syntactic
/// elements into their own nodes, so users can specify them relative to a node,
/// instead.
enum class NodePart {
  /// The node itself.
  Node,
  /// Given a \c MemberExpr, selects the member's token.
  Member,
  /// Given a \c NamedDecl or \c CxxCtorInitializer, selects that token of the
  /// relevant name, not including qualifiers.
  Name,
};

using TextGenerator =
    std::function<std::string(const ast_matchers::MatchFinder::MatchResult &)>;

/// Description of a source-code transformation.
//
// A *rewrite rule* describes a transformation of source code. It has the
// following components:
//
// * Matcher: the pattern term, expressed as clang matchers (with Transformer
//   extensions).
//
// * Target: the source code impacted by the rule. This identifies an AST node,
//   or part thereof (\c TargetPart), whose source range indicates the extent of
//   the replacement applied by the replacement term.  By default, the extent is
//   the node matched by the pattern term (\c NodePart::Node). Target's are
//   typed (\c TargetKind), which guides the determination of the node extent
//   and might, in the future, statically constrain the set of eligible
//   NodeParts for a given node.
//
// * Replacement: a function that produces a replacement string for the target,
//   based on the match result.
//
// * Explanation: explanation of the rewrite.  This will be displayed to the
//   user, where possible (for example, in clang-tidy fix descriptions).
//
// Rules have an additional, implicit, component: the parameters. These are
// portions of the pattern which are left unspecified, yet named so that we can
// reference them in the replacement term.  The structure of parameters can be
// partially or even fully specified, in which case they serve just to identify
// matched nodes for later reference rather than abstract over portions of the
// AST.  However, in all cases, we refer to named portions of the pattern as
// parameters.
//
// RewriteRule is constructed in a "fluent" style, by creating a builder and
// chaining setters of individual components.
// \code
//   RewriteRule MyRule = buildRule(functionDecl(...)).replaceWith(...);
// \endcode
//
// The \c Transformer class should then be used to apply the rewrite rule and
// obtain the corresponding replacements.
struct RewriteRule {
  // `Matcher` describes the context of this rule. It should always be bound to
  // at least `RootId`.  The builder class below takes care of this
  // binding. Here, we bind it to a trivial Matcher to enable the default
  // constructor, since DynTypedMatcher has no default constructor.
  ast_matchers::internal::DynTypedMatcher Matcher = ast_matchers::stmt();
  // The (bound) id of the node whose source will be replaced.  This id should
  // never be the empty string.
  std::string Target;
  ast_type_traits::ASTNodeKind TargetKind;
  NodePart TargetPart;
  TextGenerator Replacement;
  TextGenerator Explanation;

  // Id used as the default target of each match. The node described by the
  // matcher is guaranteed to be bound to this id, for all rewrite rules
  // constructed with the builder class.
  static constexpr llvm::StringLiteral RootId = "___root___";
};

/// A fluent builder class for \c RewriteRule.  See comments on \c RewriteRule.
class RewriteRuleBuilder {
  RewriteRule Rule;

public:
  RewriteRuleBuilder(ast_matchers::internal::DynTypedMatcher M) {
    M.setAllowBind(true);
    // `tryBind` is guaranteed to succeed, because `AllowBind` was set to true.
    Rule.Matcher = *M.tryBind(RewriteRule::RootId);
    Rule.Target = RewriteRule::RootId;
    Rule.TargetKind = M.getSupportedKind();
    Rule.TargetPart = NodePart::Node;
  }

  /// (Implicit) "build" operator to build a RewriteRule from this builder.
  operator RewriteRule() && { return std::move(Rule); }

  // Sets the target kind based on a clang AST node type.
  template <typename T> RewriteRuleBuilder as();

  template <typename T>
  RewriteRuleBuilder change(llvm::StringRef Target,
                            NodePart Part = NodePart::Node);

  RewriteRuleBuilder replaceWith(TextGenerator Replacement);
  RewriteRuleBuilder replaceWith(std::string Replacement) {
    return replaceWith(text(std::move(Replacement)));
  }

  RewriteRuleBuilder because(TextGenerator Explanation);
  RewriteRuleBuilder because(std::string Explanation) {
    return because(text(std::move(Explanation)));
  }

private:
  // Wraps a string as a TextGenerator.
  static TextGenerator text(std::string M) {
    return [M](const ast_matchers::MatchFinder::MatchResult &) { return M; };
   }
};

/// Convenience factory functions for starting construction of a \c RewriteRule.
inline RewriteRuleBuilder buildRule(ast_matchers::internal::DynTypedMatcher M) {
  return RewriteRuleBuilder(std::move(M));
}

template <typename T> RewriteRuleBuilder RewriteRuleBuilder::as() {
  Rule.TargetKind = ast_type_traits::ASTNodeKind::getFromNodeKind<T>();
  return *this;
}

template <typename T>
RewriteRuleBuilder RewriteRuleBuilder::change(llvm::StringRef TargetId,
                                              NodePart Part) {
  Rule.Target = TargetId;
  Rule.TargetKind = ast_type_traits::ASTNodeKind::getFromNodeKind<T>();
  Rule.TargetPart = Part;
  return *this;
}

/// A source "transformation," represented by a character range in the source to
/// be replaced and a corresponding replacement string.
struct Transformation {
  CharSourceRange Range;
  std::string Replacement;
};

/// Attempts to apply a rule to a match.  Returns an empty transformation if the
/// match is not eligible for rewriting (certain interactions with macros, for
/// example).  Fails if any invariants are violated relating to bound nodes in
/// the match.
Expected<Transformation>
applyRewriteRule(const RewriteRule &Rule,
                 const ast_matchers::MatchFinder::MatchResult &Match);

/// Handles the matcher and callback registration for a single rewrite rule, as
/// defined by the arguments of the constructor.
class Transformer : public ast_matchers::MatchFinder::MatchCallback {
public:
  using ChangeConsumer =
      std::function<void(const clang::tooling::AtomicChange &Change)>;

  /// \param Consumer Receives each successful rewrites as an \c AtomicChange.
  Transformer(RewriteRule Rule, ChangeConsumer Consumer)
      : Rule(std::move(Rule)), Consumer(std::move(Consumer)) {}

  /// N.B. Passes `this` pointer to `MatchFinder`.  So, this object should not
  /// be moved after this call.
  void registerMatchers(ast_matchers::MatchFinder *MatchFinder);

  /// Not called directly by users -- called by the framework, via base class
  /// pointer.
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  RewriteRule Rule;
  /// Receives each successful rewrites as an \c AtomicChange.
  ChangeConsumer Consumer;
};
} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_TRANSFORMER_H_
