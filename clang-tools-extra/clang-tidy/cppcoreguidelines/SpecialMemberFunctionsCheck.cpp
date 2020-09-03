//===--- SpecialMemberFunctionsCheck.cpp - clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

SpecialMemberFunctionsCheck::SpecialMemberFunctionsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), AllowMissingMoveFunctions(Options.get(
                                         "AllowMissingMoveFunctions", false)),
      AllowSoleDefaultDtor(Options.get("AllowSoleDefaultDtor", false)),
      AllowMissingMoveFunctionsWhenCopyIsDeleted(
          Options.get("AllowMissingMoveFunctionsWhenCopyIsDeleted", false)) {}

void SpecialMemberFunctionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowMissingMoveFunctions", AllowMissingMoveFunctions);
  Options.store(Opts, "AllowSoleDefaultDtor", AllowSoleDefaultDtor);
  Options.store(Opts, "AllowMissingMoveFunctionsWhenCopyIsDeleted",
                AllowMissingMoveFunctionsWhenCopyIsDeleted);
}

void SpecialMemberFunctionsCheck::registerMatchers(MatchFinder *Finder) {
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
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::
      DefaultDestructor:
    return "a default destructor";
  case SpecialMemberFunctionsCheck::SpecialMemberFunctionKind::
      NonDefaultDestructor:
    return "a non-default destructor";
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
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXRecordDecl>("class-def");
  if (!MatchedDecl)
    return;

  ClassDefId ID(MatchedDecl->getLocation(), std::string(MatchedDecl->getName()));

  auto StoreMember = [this, &ID](SpecialMemberFunctionData data) {
    llvm::SmallVectorImpl<SpecialMemberFunctionData> &Members =
        ClassWithSpecialMembers[ID];
    if (!llvm::is_contained(Members, data))
      Members.push_back(std::move(data));
  };

  if (const auto *Dtor = Result.Nodes.getNodeAs<CXXMethodDecl>("dtor")) {
    StoreMember({Dtor->isDefaulted()
                     ? SpecialMemberFunctionKind::DefaultDestructor
                     : SpecialMemberFunctionKind::NonDefaultDestructor,
                 Dtor->isDeleted()});
  }

  std::initializer_list<std::pair<std::string, SpecialMemberFunctionKind>>
      Matchers = {{"copy-ctor", SpecialMemberFunctionKind::CopyConstructor},
                  {"copy-assign", SpecialMemberFunctionKind::CopyAssignment},
                  {"move-ctor", SpecialMemberFunctionKind::MoveConstructor},
                  {"move-assign", SpecialMemberFunctionKind::MoveAssignment}};

  for (const auto &KV : Matchers)
    if (const auto *MethodDecl =
            Result.Nodes.getNodeAs<CXXMethodDecl>(KV.first)) {
      StoreMember({KV.second, MethodDecl->isDeleted()});
    }
}

void SpecialMemberFunctionsCheck::onEndOfTranslationUnit() {
  for (const auto &C : ClassWithSpecialMembers) {
    checkForMissingMembers(C.first, C.second);
  }
}

void SpecialMemberFunctionsCheck::checkForMissingMembers(
    const ClassDefId &ID,
    llvm::ArrayRef<SpecialMemberFunctionData> DefinedMembers) {
  llvm::SmallVector<SpecialMemberFunctionKind, 5> MissingMembers;

  auto HasMember = [&](SpecialMemberFunctionKind Kind) {
    return llvm::any_of(DefinedMembers, [Kind](const auto &data) {
      return data.FunctionKind == Kind;
    });
  };

  auto IsDeleted = [&](SpecialMemberFunctionKind Kind) {
    return llvm::any_of(DefinedMembers, [Kind](const auto &data) {
      return data.FunctionKind == Kind && data.IsDeleted;
    });
  };

  auto RequireMember = [&](SpecialMemberFunctionKind Kind) {
    if (!HasMember(Kind))
      MissingMembers.push_back(Kind);
  };

  bool RequireThree =
      HasMember(SpecialMemberFunctionKind::NonDefaultDestructor) ||
      (!AllowSoleDefaultDtor &&
       HasMember(SpecialMemberFunctionKind::DefaultDestructor)) ||
      HasMember(SpecialMemberFunctionKind::CopyConstructor) ||
      HasMember(SpecialMemberFunctionKind::CopyAssignment) ||
      HasMember(SpecialMemberFunctionKind::MoveConstructor) ||
      HasMember(SpecialMemberFunctionKind::MoveAssignment);

  bool RequireFive = (!AllowMissingMoveFunctions && RequireThree &&
                      getLangOpts().CPlusPlus11) ||
                     HasMember(SpecialMemberFunctionKind::MoveConstructor) ||
                     HasMember(SpecialMemberFunctionKind::MoveAssignment);

  if (RequireThree) {
    if (!HasMember(SpecialMemberFunctionKind::DefaultDestructor) &&
        !HasMember(SpecialMemberFunctionKind::NonDefaultDestructor))
      MissingMembers.push_back(SpecialMemberFunctionKind::Destructor);

    RequireMember(SpecialMemberFunctionKind::CopyConstructor);
    RequireMember(SpecialMemberFunctionKind::CopyAssignment);
  }

  if (RequireFive &&
      !(AllowMissingMoveFunctionsWhenCopyIsDeleted &&
        (IsDeleted(SpecialMemberFunctionKind::CopyConstructor) &&
         IsDeleted(SpecialMemberFunctionKind::CopyAssignment)))) {
    assert(RequireThree);
    RequireMember(SpecialMemberFunctionKind::MoveConstructor);
    RequireMember(SpecialMemberFunctionKind::MoveAssignment);
  }

  if (!MissingMembers.empty()) {
    llvm::SmallVector<SpecialMemberFunctionKind, 5> DefinedMemberKinds;
    llvm::transform(DefinedMembers, std::back_inserter(DefinedMemberKinds),
                    [](const auto &data) { return data.FunctionKind; });
    diag(ID.first, "class '%0' defines %1 but does not define %2")
        << ID.second << cppcoreguidelines::join(DefinedMemberKinds, " and ")
        << cppcoreguidelines::join(MissingMembers, " or ");
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
