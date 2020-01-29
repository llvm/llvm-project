//===--- RewriteRule.h - RewriteRule class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file
///  Defines the RewriteRule class and related functions for creating,
///  modifying and interpreting RewriteRules.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_TRANSFORMER_REWRITE_RULE_H_
#define LLVM_CLANG_TOOLING_TRANSFORMER_REWRITE_RULE_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Transformer/MatchConsumer.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <string>
#include <utility>

namespace clang {
namespace transformer {
using TextGenerator = std::shared_ptr<MatchComputation<std::string>>;

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
// `ASTEdit` should be built using the `change` convenience functions. For
// example,
// \code
//   changeTo(name(fun), cat("Frodo"))
// \endcode
// Or, if we use Stencil for the TextGenerator:
// \code
//   using stencil::cat;
//   changeTo(statement(thenNode), cat("{", thenNode, "}"))
//   changeTo(callArgs(call), cat(x, ",", y))
// \endcode
// Or, if you are changing the node corresponding to the rule's matcher, you can
// use the single-argument override of \c change:
// \code
//   changeTo(cat("different_expr"))
// \endcode
struct ASTEdit {
  RangeSelector TargetRange;
  TextGenerator Replacement;
  TextGenerator Note;
};

/// Format of the path in an include directive -- angle brackets or quotes.
enum class IncludeFormat {
  Quoted,
  Angled,
};

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
    // Include paths to add to the file affected by this case.  These are
    // bundled with the `Case`, rather than the `RewriteRule`, because each case
    // might have different associated changes to the includes.
    std::vector<std::pair<std::string, IncludeFormat>> AddedIncludes;
  };
  // We expect RewriteRules will most commonly include only one case.
  SmallVector<Case, 1> Cases;

  // ID used as the default target of each match. The node described by the
  // matcher is should always be bound to this id.
  static constexpr llvm::StringLiteral RootID = "___root___";
};

/// Convenience function for constructing a simple \c RewriteRule.
RewriteRule makeRule(ast_matchers::internal::DynTypedMatcher M,
                     SmallVector<ASTEdit, 1> Edits,
                     TextGenerator Explanation = nullptr);

/// Convenience overload of \c makeRule for common case of only one edit.
inline RewriteRule makeRule(ast_matchers::internal::DynTypedMatcher M,
                            ASTEdit Edit,
                            TextGenerator Explanation = nullptr) {
  SmallVector<ASTEdit, 1> Edits;
  Edits.emplace_back(std::move(Edit));
  return makeRule(std::move(M), std::move(Edits), std::move(Explanation));
}

/// For every case in Rule, adds an include directive for the given header. The
/// common use is assumed to be a rule with only one case. For example, to
/// replace a function call and add headers corresponding to the new code, one
/// could write:
/// \code
///   auto R = makeRule(callExpr(callee(functionDecl(hasName("foo")))),
///            changeTo(cat("bar()")));
///   AddInclude(R, "path/to/bar_header.h");
///   AddInclude(R, "vector", IncludeFormat::Angled);
/// \endcode
void addInclude(RewriteRule &Rule, llvm::StringRef Header,
                IncludeFormat Format = IncludeFormat::Quoted);

/// Applies the first rule whose pattern matches; other rules are ignored.  If
/// the matchers are independent then order doesn't matter. In that case,
/// `applyFirst` is simply joining the set of rules into one.
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

/// Replaces a portion of the source text with \p Replacement.
ASTEdit changeTo(RangeSelector Target, TextGenerator Replacement);
/// DEPRECATED: use \c changeTo.
inline ASTEdit change(RangeSelector Target, TextGenerator Replacement) {
  return changeTo(std::move(Target), std::move(Replacement));
}

/// Replaces the entirety of a RewriteRule's match with \p Replacement.  For
/// example, to replace a function call, one could write:
/// \code
///   makeRule(callExpr(callee(functionDecl(hasName("foo")))),
///            changeTo(cat("bar()")))
/// \endcode
inline ASTEdit changeTo(TextGenerator Replacement) {
  return changeTo(node(std::string(RewriteRule::RootID)),
                  std::move(Replacement));
}
/// DEPRECATED: use \c changeTo.
inline ASTEdit change(TextGenerator Replacement) {
  return changeTo(std::move(Replacement));
}

/// Inserts \p Replacement before \p S, leaving the source selected by \S
/// unchanged.
inline ASTEdit insertBefore(RangeSelector S, TextGenerator Replacement) {
  return changeTo(before(std::move(S)), std::move(Replacement));
}

/// Inserts \p Replacement after \p S, leaving the source selected by \S
/// unchanged.
inline ASTEdit insertAfter(RangeSelector S, TextGenerator Replacement) {
  return changeTo(after(std::move(S)), std::move(Replacement));
}

/// Removes the source selected by \p S.
ASTEdit remove(RangeSelector S);

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
/// Only supports Rules whose cases' matchers share the same base "kind"
/// (`Stmt`, `Decl`, etc.)  Deprecated: use `buildMatchers` instead, which
/// supports mixing matchers of different kinds.
ast_matchers::internal::DynTypedMatcher buildMatcher(const RewriteRule &Rule);

/// Builds a set of matchers that cover the rule (one for each distinct node
/// matcher base kind: Stmt, Decl, etc.). Node-matchers for `QualType` and
/// `Type` are not permitted, since such nodes carry no source location
/// information and are therefore not relevant for rewriting. If any such
/// matchers are included, will return an empty vector.
std::vector<ast_matchers::internal::DynTypedMatcher>
buildMatchers(const RewriteRule &Rule);

/// Gets the beginning location of the source matched by a rewrite rule. If the
/// match occurs within a macro expansion, returns the beginning of the
/// expansion point. `Result` must come from the matching of a rewrite rule.
SourceLocation
getRuleMatchLoc(const ast_matchers::MatchFinder::MatchResult &Result);

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
} // namespace transformer

namespace tooling {
// DEPRECATED: These are temporary aliases supporting client migration to the
// `transformer` namespace.
/// Wraps a string as a TextGenerator.
using TextGenerator = transformer::TextGenerator;

TextGenerator text(std::string M);

using transformer::addInclude;
using transformer::applyFirst;
using transformer::change;
using transformer::insertAfter;
using transformer::insertBefore;
using transformer::makeRule;
using transformer::remove;
using transformer::RewriteRule;
using transformer::IncludeFormat;
namespace detail {
using namespace transformer::detail;
} // namespace detail
} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_TRANSFORMER_REWRITE_RULE_H_
