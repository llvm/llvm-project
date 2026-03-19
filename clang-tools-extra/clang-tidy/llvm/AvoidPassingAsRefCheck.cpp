//===--- AvoidPassingAsRefCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPassingAsRefCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

AvoidPassingAsRefCheck::AvoidPassingAsRefCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ClassNameList(utils::options::parseStringList(
          Options.get("ClassNames", "::mlir::Op"))) {}

void AvoidPassingAsRefCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ClassNames", ClassNames);
}

void AvoidPassingAsRefCheck::registerMatchers(MatchFinder *Finder) {
  if (ClassNameList.empty())
    return;

  std::vector<ast_matchers::internal::Matcher<CXXRecordDecl>> Matchers;
  for (const auto &Name : ClassNameList) {
    Matchers.push_back(isSameOrDerivedFrom(std::string(Name)));
  }

  // Match parameters that are references to classes derived from any class in
  // ClassNameList.
  Finder->addMatcher(
      parmVarDecl(
          hasType(qualType(
              references(qualType(hasDeclaration(
                  cxxRecordDecl(anyOfArrayRef(Matchers)).bind("op_type")))),
              // We want to avoid matching `Op *&` (reference to pointer to Op)
              // which is not common for Op but possible.
              unless(references(pointerType())))),
          unless(isImplicit()),
          unless(hasAncestor(cxxConstructorDecl(
              anyOf(isCopyConstructor(), isMoveConstructor())))),
          unless(hasAncestor(cxxMethodDecl(isCopyAssignmentOperator()))),
          unless(hasAncestor(cxxMethodDecl(isMoveAssignmentOperator()))),
          decl().bind("param")),
      this);
}

void AvoidPassingAsRefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *OpType = Result.Nodes.getNodeAs<CXXRecordDecl>("op_type");

  // Exclude if we can't find definitions.
  if (!Param || !OpType)
    return;

  // We should verify if the type is exactly what we expect. The matcher
  // `isSameOrDerivedFrom` handles inheritance.
  diag(Param->getLocation(),
       "class '%0' should be passed by value, not by reference")
      << OpType->getName();
}

} // namespace clang::tidy::llvm_check
