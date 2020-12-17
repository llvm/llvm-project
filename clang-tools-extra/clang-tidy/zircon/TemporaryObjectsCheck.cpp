//===--- TemporaryObjectsCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TemporaryObjectsCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace zircon {

AST_MATCHER_P(CXXRecordDecl, matchesAnyName, ArrayRef<std::string>, Names) {
  std::string QualifiedName = Node.getQualifiedNameAsString();
  return llvm::any_of(Names,
                      [&](StringRef Name) { return QualifiedName == Name; });
}

void TemporaryObjectsCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for default constructors.
  Finder->addMatcher(
      cxxTemporaryObjectExpr(hasDeclaration(cxxConstructorDecl(hasParent(
                                 cxxRecordDecl(matchesAnyName(Names))))))
          .bind("temps"),
      this);

  // Matcher for user-defined constructors.
  Finder->addMatcher(
      traverse(TK_AsIs,
               cxxConstructExpr(hasParent(cxxFunctionalCastExpr()),
                                hasDeclaration(cxxConstructorDecl(hasParent(
                                    cxxRecordDecl(matchesAnyName(Names))))))
                   .bind("temps")),
      this);
}

void TemporaryObjectsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<CXXConstructExpr>("temps"))
    diag(D->getLocation(),
         "creating a temporary object of type %q0 is prohibited")
        << D->getConstructor()->getParent();
}

void TemporaryObjectsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Names", utils::options::serializeStringList(Names));
}

} // namespace zircon
} // namespace tidy
} // namespace clang
