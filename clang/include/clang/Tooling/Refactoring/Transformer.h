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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <deque>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

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

// Note that \p TextGenerator is allowed to fail, e.g. when trying to access a
// matched node that was not bound.  Allowing this to fail simplifies error
// handling for interactive tools like clang-query.
using TextGenerator = std::function<Expected<std::string>(
    const ast_matchers::MatchFinder::MatchResult &)>;

/// Wraps a string as a TextGenerator.
inline TextGenerator text(std::string M) {
  return [M](const ast_matchers::MatchFinder::MatchResult &)
             -> Expected<std::string> { return M; };
}

// Description of a source-code edit, expressed in terms of an AST node.
// Includes: an ID for the (bound) node, a selector for source related to the
// node, a replacement and, optionally, an explanation for the edit.
//
// * Target: the source code impacted by the rule. This identifies an AST node,
//   or part thereof (\c Part), whose source range indicates the extent of the
//   replacement applied by the replacement term.  By default, the extent is the
//   node matched by the pattern term (\c NodePart::Node). Target's are typed
//   (\c Kind), which guides the determination of the node extent.
//
// * Replacement: a function that produces a replacement string for the target,
//   based on the match result.
//
// * Note: (optional) a note specifically for this edit, potentially referencing
//   elements of the match.  This will be displayed to the user, where possible;
//   for example, in clang-tidy diagnostics.  Use of notes should be rare --
//   explanations of the entire rewrite should be set in the rule
//   (`RewriteRule::Explanation`) instead.  Notes serve the rare cases wherein
//   edit-specific diagnostics are required.
//
// `ASTEdit` should be built using the `change` convenience fucntions. For
// example,
// \code
//   change<FunctionDecl>(fun, NodePart::Name, "Frodo")
// \endcode
// Or, if we use Stencil for the TextGenerator:
// \code
//   change<Stmt>(thenNode, stencil::cat("{", thenNode, "}"))
//   change<Expr>(call, NodePart::Args, stencil::cat(x, ",", y))
//     .note("argument order changed.")
// \endcode
// Or, if you are changing the node corresponding to the rule's matcher, you can
// use the single-argument override of \c change:
// \code
//   change<Expr>("different_expr")
// \endcode
struct ASTEdit {
  // The (bound) id of the node whose source will be replaced.  This id should
  // never be the empty string.
  std::string Target;
  ast_type_traits::ASTNodeKind Kind;
  NodePart Part;
  TextGenerator Replacement;
  TextGenerator Note;
};

// Convenience functions for creating \c ASTEdits.  They all must be explicitly
// instantiated with the desired AST type.  Each overload includes both \c
// std::string and \c TextGenerator versions.

// FIXME: For overloads taking a \c NodePart, constrain the valid values of \c
// Part based on the type \c T.
template <typename T>
ASTEdit change(StringRef Target, NodePart Part, TextGenerator Replacement) {
  ASTEdit E;
  E.Target = Target.str();
  E.Kind = ast_type_traits::ASTNodeKind::getFromNodeKind<T>();
  E.Part = Part;
  E.Replacement = std::move(Replacement);
  return E;
}

template <typename T>
ASTEdit change(StringRef Target, NodePart Part, std::string Replacement) {
  return change<T>(Target, Part, text(std::move(Replacement)));
}

/// Variant of \c change for which the NodePart defaults to the whole node.
template <typename T>
ASTEdit change(StringRef Target, TextGenerator Replacement) {
  return change<T>(Target, NodePart::Node, std::move(Replacement));
}

/// Variant of \c change for which the NodePart defaults to the whole node.
template <typename T>
ASTEdit change(StringRef Target, std::string Replacement) {
  return change<T>(Target, text(std::move(Replacement)));
}

/// Variant of \c change that selects the node of the entire match.
template <typename T> ASTEdit change(TextGenerator Replacement);

/// Variant of \c change that selects the node of the entire match.
template <typename T> ASTEdit change(std::string Replacement) {
  return change<T>(text(std::move(Replacement)));
}

/// Description of a source-code transformation.
//
// A *rewrite rule* describes a transformation of source code. A simple rule
// contains each of the following components:
//
// * Matcher: the pattern term, expressed as clang matchers (with Transformer
//   extensions).
//
// * Edits: a set of Edits to the source code, described with ASTEdits.
//
// * Explanation: explanation of the rewrite.  This will be displayed to the
//   user, where possible; for example, in clang-tidy diagnostics.
//
// However, rules can also consist of (sub)rules, where the first that matches
// is applied and the rest are ignored.  So, the above components are gathered
// as a `Case` and a rule is a list of cases.
//
// Rule cases have an additional, implicit, component: the parameters. These are
// portions of the pattern which are left unspecified, yet bound in the pattern
// so that we can reference them in the edits.
//
// The \c Transformer class can be used to apply the rewrite rule and obtain the
// corresponding replacements.
struct RewriteRule {
  struct Case {
    ast_matchers::internal::DynTypedMatcher Matcher;
    SmallVector<ASTEdit, 1> Edits;
    TextGenerator Explanation;
  };
  // We expect RewriteRules will most commonly include only one case.
  SmallVector<Case, 1> Cases;

  // Id used as the default target of each match. The node described by the
  // matcher is should always be bound to this id.
  static constexpr llvm::StringLiteral RootId = "___root___";
};

/// Convenience function for constructing a simple \c RewriteRule.
RewriteRule makeRule(ast_matchers::internal::DynTypedMatcher M,
                     SmallVector<ASTEdit, 1> Edits);

/// Convenience overload of \c makeRule for common case of only one edit.
inline RewriteRule makeRule(ast_matchers::internal::DynTypedMatcher M,
                            ASTEdit Edit) {
  SmallVector<ASTEdit, 1> Edits;
  Edits.emplace_back(std::move(Edit));
  return makeRule(std::move(M), std::move(Edits));
}

/// Applies the first rule whose pattern matches; other rules are ignored.
///
/// N.B. All of the rules must use the same kind of matcher (that is, share a
/// base class in the AST hierarchy).  However, this constraint is caused by an
/// implementation detail and should be lifted in the future.
//
// `applyFirst` is like an `anyOf` matcher with an edit action attached to each
// of its cases. Anywhere you'd use `anyOf(m1.bind("id1"), m2.bind("id2"))` and
// then dispatch on those ids in your code for control flow, `applyFirst` lifts
// that behavior to the rule level.  So, you can write `applyFirst({makeRule(m1,
// action1), makeRule(m2, action2), ...});`
//
// For example, consider a type `T` with a deterministic serialization function,
// `serialize()`.  For performance reasons, we would like to make it
// non-deterministic.  Therefore, we want to drop the expectation that
// `a.serialize() = b.serialize() iff a = b` (although we'll maintain
// `deserialize(a.serialize()) = a`).
//
// We have three cases to consider (for some equality function, `eq`):
// ```
// eq(a.serialize(), b.serialize()) --> eq(a,b)
// eq(a, b.serialize())             --> eq(deserialize(a), b)
// eq(a.serialize(), b)             --> eq(a, deserialize(b))
// ```
//
// `applyFirst` allows us to specify each independently:
// ```
// auto eq_fun = functionDecl(...);
// auto method_call = cxxMemberCallExpr(...);
//
// auto two_calls = callExpr(callee(eq_fun), hasArgument(0, method_call),
//                           hasArgument(1, method_call));
// auto left_call =
//     callExpr(callee(eq_fun), callExpr(hasArgument(0, method_call)));
// auto right_call =
//     callExpr(callee(eq_fun), callExpr(hasArgument(1, method_call)));
//
// RewriteRule R = applyFirst({makeRule(two_calls, two_calls_action),
//                             makeRule(left_call, left_call_action),
//                             makeRule(right_call, right_call_action)});
// ```
RewriteRule applyFirst(ArrayRef<RewriteRule> Rules);

// Define this overload of `change` here because RewriteRule::RootId is not in
// scope at the declaration point above.
template <typename T> ASTEdit change(TextGenerator Replacement) {
  return change<T>(RewriteRule::RootId, NodePart::Node, std::move(Replacement));
}

/// The following three functions are a low-level part of the RewriteRule
/// API. We expose them for use in implementing the fixtures that interpret
/// RewriteRule, like Transformer and TransfomerTidy, or for more advanced
/// users.
//
// FIXME: These functions are really public, if advanced, elements of the
// RewriteRule API.  Recast them as such.  Or, just declare these functions
// public and well-supported and move them out of `detail`.
namespace detail {
/// Builds a single matcher for the rule, covering all of the rule's cases.
ast_matchers::internal::DynTypedMatcher buildMatcher(const RewriteRule &Rule);

/// Returns the \c Case of \c Rule that was selected in the match result.
/// Assumes a matcher built with \c buildMatcher.
const RewriteRule::Case &
findSelectedCase(const ast_matchers::MatchFinder::MatchResult &Result,
                 const RewriteRule &Rule);

/// A source "transformation," represented by a character range in the source to
/// be replaced and a corresponding replacement string.
struct Transformation {
  CharSourceRange Range;
  std::string Replacement;
};

/// Attempts to translate `Edits`, which are in terms of AST nodes bound in the
/// match `Result`, into Transformations, which are in terms of the source code
/// text.
///
/// Returns an empty vector if any of the edits apply to portions of the source
/// that are ineligible for rewriting (certain interactions with macros, for
/// example).  Fails if any invariants are violated relating to bound nodes in
/// the match.  However, it does not fail in the case of conflicting edits --
/// conflict handling is left to clients.  We recommend use of the \c
/// AtomicChange or \c Replacements classes for assistance in detecting such
/// conflicts.
Expected<SmallVector<Transformation, 1>>
translateEdits(const ast_matchers::MatchFinder::MatchResult &Result,
               llvm::ArrayRef<ASTEdit> Edits);
} // namespace detail

/// Handles the matcher and callback registration for a single rewrite rule, as
/// defined by the arguments of the constructor.
class Transformer : public ast_matchers::MatchFinder::MatchCallback {
public:
  using ChangeConsumer =
      std::function<void(Expected<clang::tooling::AtomicChange> Change)>;

  /// \param Consumer Receives each rewrite or error.  Will not necessarily be
  /// called for each match; for example, if the rewrite is not applicable
  /// because of macros, but doesn't fail.  Note that clients are responsible
  /// for handling the case that independent \c AtomicChanges conflict with each
  /// other.
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
