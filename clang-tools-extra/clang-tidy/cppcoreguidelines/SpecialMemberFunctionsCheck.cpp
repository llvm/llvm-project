//===--- SpecialMemberFunctionsCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SpecialMemberFunctionsCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "clang-tidy"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

void SpecialMemberFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;
  Finder->addMatcher(
      cxxRecordDecl(
          eachOf(
              has(cxxDestructorDecl(unless(isImplicit())).bind("dtor")),
              has(cxxConstructorDecl(isCopyConstructor(), unless(isImplicit()))
                      .bind("copy-ctor")),
              has(cxxMethodDecl(isCopyAssignmentOperator(),
                                unless(isImplicit()))
                      .bind("copy-assign")),
              has(cxxConstructorDecl(isMoveConstructor(), unless(isImplicit()))
                      .bind("move-ctor")),
              has(cxxMethodDecl(isMoveAssignmentOperator(),
                                unless(isImplicit()))
                      .bind("move-assign"))))
          .bind("class-def"),
      this);
}

static llvm::StringRef
toString(SpecialMemberFunctionsCheck::SpecialMemberFunctionKind K) {
  switch (K) {
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::Destructor:
    return "a destructor";
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::CopyConstructor:
    return "a copy constructor";
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::CopyAssignment:
    return "a copy assignment operator";
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::MoveConstructor:
    return "a move constructor";
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::MoveAssignment:
    return "a move assignment operator";
  }
  llvm_unreachable("Unhandled SpecialMemberFunctionKind");
}

static std::string
join(ArrayRef<SpecialMemberFunctionsCheck::SpecialMemberFunctionKind> SMFS,
     llvm::StringRef AndOr) {

  assert(!SMFS.empty() &&
         "List of defined or undefined members should never be empty.");
  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);

  Stream << toString(SMFS[0]);
  size_t LastIndex = SMFS.size() - 1;
  for (size_t i = 1; i < LastIndex; ++i) {
    Stream << ", " << toString(SMFS[i]);
  }
  if (LastIndex != 0) {
    Stream << AndOr << toString(SMFS[LastIndex]);
  }
  return Stream.str();
}

void SpecialMemberFunctionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const CXXRecordDecl *MatchedDecl =
      Result.Nodes.getNodeAs<CXXRecordDecl>("class-def");
  if (!MatchedDecl)
    return;

  ClassDefId ID(MatchedDecl->getLocation(), MatchedDecl->getName());

  std::initializer_list<std::pair<std::string, SpecialMemberFunctionKind>>
      Matchers = {{"dtor", SpecialMemberFunctionKind::Destructor},
                  {"copy-ctor", SpecialMemberFunctionKind::CopyConstructor},
                  {"copy-assign", SpecialMemberFunctionKind::CopyAssignment},
                  {"move-ctor", SpecialMemberFunctionKind::MoveConstructor},
                  {"move-assign", SpecialMemberFunctionKind::MoveAssignment}};

  for (const auto &KV : Matchers)
    if (Result.Nodes.getNodeAs<CXXMethodDecl>(KV.first))
      ClassWithSpecialMembers[ID].insert(KV.second);
}

void SpecialMemberFunctionsCheck::onEndOfTranslationUnit() {
  llvm::SmallVector<SpecialMemberFunctionKind, 5> AllSpecialMembers = {
      SpecialMemberFunctionKind::Destructor,
      SpecialMemberFunctionKind::CopyConstructor,
      SpecialMemberFunctionKind::CopyAssignment};

  if (getLangOpts().CPlusPlus11) {
    AllSpecialMembers.push_back(SpecialMemberFunctionKind::MoveConstructor);
    AllSpecialMembers.push_back(SpecialMemberFunctionKind::MoveAssignment);
  }

  for (const auto &C : ClassWithSpecialMembers) {
    const auto &DefinedSpecialMembers = C.second;

    if (DefinedSpecialMembers.size() == AllSpecialMembers.size())
      continue;

    llvm::SmallVector<SpecialMemberFunctionKind, 5> UndefinedSpecialMembers;
    std::set_difference(AllSpecialMembers.begin(), AllSpecialMembers.end(),
                        DefinedSpecialMembers.begin(),
                        DefinedSpecialMembers.end(),
                        std::back_inserter(UndefinedSpecialMembers));

    diag(C.first.first, "class '%0' defines %1 but does not define %2")
        << C.first.second << join(DefinedSpecialMembers.getArrayRef(), " and ")
        << join(UndefinedSpecialMembers, " or ");
  }
}
} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
