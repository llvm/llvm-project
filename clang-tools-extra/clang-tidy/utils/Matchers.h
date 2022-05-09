//===--- Matchers.h - clang-tidy-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H

#include "TypeTraits.h"
#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace tidy {
namespace matchers {

AST_MATCHER(BinaryOperator, isRelationalOperator) {
  return Node.isRelationalOp();
}

AST_MATCHER(BinaryOperator, isEqualityOperator) { return Node.isEqualityOp(); }

AST_MATCHER(QualType, isExpensiveToCopy) {
  llvm::Optional<bool> IsExpensive =
      utils::type_traits::isExpensiveToCopy(Node, Finder->getASTContext());
  return IsExpensive && *IsExpensive;
}

AST_MATCHER(RecordDecl, isTriviallyDefaultConstructible) {
  return utils::type_traits::recordIsTriviallyDefaultConstructible(
      Node, Finder->getASTContext());
}

AST_MATCHER(QualType, isTriviallyDestructible) {
  return utils::type_traits::isTriviallyDestructible(Node);
}

// Returns QualType matcher for references to const.
AST_MATCHER_FUNCTION(ast_matchers::TypeMatcher, isReferenceToConst) {
  using namespace ast_matchers;
  return referenceType(pointee(qualType(isConstQualified())));
}

// Returns QualType matcher for pointers to const.
AST_MATCHER_FUNCTION(ast_matchers::TypeMatcher, isPointerToConst) {
  using namespace ast_matchers;
  return pointerType(pointee(qualType(isConstQualified())));
}

// A matcher implementation that matches a list of type name regular expressions
// against a NamedDecl. If a regular expression contains the substring "::"
// matching will occur against the qualified name, otherwise only the typename.
class MatchesAnyListedNameMatcher
    : public ast_matchers::internal::MatcherInterface<NamedDecl> {
public:
  explicit MatchesAnyListedNameMatcher(llvm::ArrayRef<StringRef> NameList) {
    std::transform(
        NameList.begin(), NameList.end(), std::back_inserter(NameMatchers),
        [](const llvm::StringRef Name) { return NameMatcher(Name); });
  }
  bool matches(
      const NamedDecl &Node, ast_matchers::internal::ASTMatchFinder *Finder,
      ast_matchers::internal::BoundNodesTreeBuilder *Builder) const override {
    return llvm::any_of(NameMatchers, [&Node](const NameMatcher &NM) {
      return NM.match(Node);
    });
  }

private:
  class NameMatcher {
    llvm::Regex Regex;
    enum class MatchMode {
      // Match against the unqualified name because the regular expression
      // does not contain ":".
      MatchUnqualified,
      // Match against the qualified name because the regular expression
      // contains ":" suggesting name and namespace should be matched.
      MatchQualified,
      // Match against the fully qualified name because the regular expression
      // starts with ":".
      MatchFullyQualified,
    };
    MatchMode Mode;

  public:
    NameMatcher(const llvm::StringRef Regex)
        : Regex(Regex), Mode(determineMatchMode(Regex)) {}

    bool match(const NamedDecl &ND) const {
      switch (Mode) {
      case MatchMode::MatchQualified:
        return Regex.match(ND.getQualifiedNameAsString());
      case MatchMode::MatchFullyQualified:
        return Regex.match("::" + ND.getQualifiedNameAsString());
      default:
        return Regex.match(ND.getName());
      }
    }

  private:
    MatchMode determineMatchMode(llvm::StringRef Regex) {
      if (Regex.startswith(":") || Regex.startswith("^:")) {
        return MatchMode::MatchFullyQualified;
      }
      return Regex.contains(":") ? MatchMode::MatchQualified
                                 : MatchMode::MatchUnqualified;
    }
  };

  std::vector<NameMatcher> NameMatchers;
};

// Returns a matcher that matches NamedDecl's against a list of provided regular
// expressions. If a regular expression contains starts ':' the NamedDecl's
// qualified name will be used for matching, otherwise its name will be used.
inline ::clang::ast_matchers::internal::Matcher<NamedDecl>
matchesAnyListedName(llvm::ArrayRef<StringRef> NameList) {
  return ::clang::ast_matchers::internal::makeMatcher(
      new MatchesAnyListedNameMatcher(NameList));
}

} // namespace matchers
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_MATCHERS_H
