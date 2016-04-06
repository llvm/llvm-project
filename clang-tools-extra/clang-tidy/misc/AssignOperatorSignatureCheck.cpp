//===--- AssignOperatorSignatureCheck.cpp - clang-tidy ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AssignOperatorSignatureCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

void AssignOperatorSignatureCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto HasGoodReturnType = cxxMethodDecl(returns(
      lValueReferenceType(pointee(unless(isConstQualified()),
                                  hasDeclaration(equalsBoundNode("class"))))));

  const auto IsSelf = qualType(
      anyOf(hasDeclaration(equalsBoundNode("class")),
            referenceType(pointee(hasDeclaration(equalsBoundNode("class"))))));
  const auto IsAssign =
      cxxMethodDecl(unless(anyOf(isDeleted(), isPrivate(), isImplicit())),
                    hasName("operator="), ofClass(recordDecl().bind("class")))
          .bind("method");
  const auto IsSelfAssign =
      cxxMethodDecl(IsAssign, hasParameter(0, parmVarDecl(hasType(IsSelf))))
          .bind("method");

  Finder->addMatcher(
      cxxMethodDecl(IsAssign, unless(HasGoodReturnType)).bind("ReturnType"),
      this);

  const auto BadSelf = referenceType(
      anyOf(lValueReferenceType(pointee(unless(isConstQualified()))),
            rValueReferenceType(pointee(isConstQualified()))));

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign,
                    hasParameter(0, parmVarDecl(hasType(BadSelf))))
          .bind("ArgumentType"),
      this);

  Finder->addMatcher(
      cxxMethodDecl(IsSelfAssign, anyOf(isConst(), isVirtual())).bind("cv"),
      this);
}

void AssignOperatorSignatureCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Method = Result.Nodes.getNodeAs<CXXMethodDecl>("method");
  std::string Name = Method->getParent()->getName();

  static const char *const Messages[][2] = {
      {"ReturnType", "operator=() should return '%0&'"},
      {"ArgumentType", "operator=() should take '%0 const&', '%0&&' or '%0'"},
      {"cv", "operator=() should not be marked '%1'"}};

  for (const auto &Message : Messages) {
    if (Result.Nodes.getNodeAs<Decl>(Message[0]))
      diag(Method->getLocStart(), Message[1])
          << Name << (Method->isConst() ? "const" : "virtual");
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
