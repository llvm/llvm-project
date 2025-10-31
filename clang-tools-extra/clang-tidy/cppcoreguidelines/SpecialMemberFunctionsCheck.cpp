//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecialMemberFunctionsCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringExtras.h"

#define DEBUG_TYPE "clang-tidy"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER(CXXRecordDecl, isInMacro) {
  return Node.getBeginLoc().isMacroID() && Node.getEndLoc().isMacroID();
}
} // namespace

SpecialMemberFunctionsCheck::SpecialMemberFunctionsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), AllowMissingMoveFunctions(Options.get(
                                         "AllowMissingMoveFunctions", false)),
      AllowSoleDefaultDtor(Options.get("AllowSoleDefaultDtor", false)),
      AllowMissingMoveFunctionsWhenCopyIsDeleted(
          Options.get("AllowMissingMoveFunctionsWhenCopyIsDeleted", false)),
      AllowImplicitlyDeletedCopyOrMove(
          Options.get("AllowImplicitlyDeletedCopyOrMove", false)),
      IgnoreMacros(Options.get("IgnoreMacros", true)) {}

void SpecialMemberFunctionsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowMissingMoveFunctions", AllowMissingMoveFunctions);
  Options.store(Opts, "AllowSoleDefaultDtor", AllowSoleDefaultDtor);
  Options.store(Opts, "AllowMissingMoveFunctionsWhenCopyIsDeleted",
                AllowMissingMoveFunctionsWhenCopyIsDeleted);
  Options.store(Opts, "AllowImplicitlyDeletedCopyOrMove",
                AllowImplicitlyDeletedCopyOrMove);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

std::optional<TraversalKind>
SpecialMemberFunctionsCheck::getCheckTraversalKind() const {
  return AllowImplicitlyDeletedCopyOrMove ? TK_AsIs
                                          : TK_IgnoreUnlessSpelledInSource;
}

void SpecialMemberFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsNotImplicitOrDeleted = anyOf(unless(isImplicit()), isDeleted());
  const ast_matchers::internal::Matcher<CXXRecordDecl> Anything = anything();

  Finder->addMatcher(
      cxxRecordDecl(
          unless(isImplicit()), IgnoreMacros ? unless(isInMacro()) : Anything,
          eachOf(has(cxxDestructorDecl(unless(isImplicit())).bind("dtor")),
                 has(cxxConstructorDecl(isCopyConstructor(),
                                        IsNotImplicitOrDeleted)
                         .bind("copy-ctor")),
                 has(cxxMethodDecl(isCopyAssignmentOperator(),
                                   IsNotImplicitOrDeleted)
                         .bind("copy-assign")),
                 has(cxxConstructorDecl(isMoveConstructor(),
                                        IsNotImplicitOrDeleted)
                         .bind("move-ctor")),
                 has(cxxMethodDecl(isMoveAssignmentOperator(),
                                   IsNotImplicitOrDeleted)
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
  for (size_t I = 1; I < LastIndex; ++I) {
    Stream << ", " << toString(SMFS[I]);
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

  ClassDefId ID(MatchedDecl->getLocation(),
                std::string(MatchedDecl->getName()));

  auto StoreMember = [this, &ID](SpecialMemberFunctionData Data) {
    llvm::SmallVectorImpl<SpecialMemberFunctionData> &Members =
        ClassWithSpecialMembers[ID];
    if (!llvm::is_contained(Members, Data))
      Members.push_back(std::move(Data));
  };

  if (const auto *Dtor = Result.Nodes.getNodeAs<CXXMethodDecl>("dtor")) {
    SpecialMemberFunctionKind DestructorType =
        SpecialMemberFunctionKind::Destructor;
    if (Dtor->isDefined()) {
      DestructorType = Dtor->getDefinition()->isDefaulted()
                           ? SpecialMemberFunctionKind::DefaultDestructor
                           : SpecialMemberFunctionKind::NonDefaultDestructor;
    }
    StoreMember({DestructorType, Dtor->isDeleted()});
  }

  std::initializer_list<std::pair<std::string, SpecialMemberFunctionKind>>
      Matchers = {{"copy-ctor", SpecialMemberFunctionKind::CopyConstructor},
                  {"copy-assign", SpecialMemberFunctionKind::CopyAssignment},
                  {"move-ctor", SpecialMemberFunctionKind::MoveConstructor},
                  {"move-assign", SpecialMemberFunctionKind::MoveAssignment}};

  for (const auto &KV : Matchers)
    if (const auto *MethodDecl =
            Result.Nodes.getNodeAs<CXXMethodDecl>(KV.first)) {
      StoreMember(
          {KV.second, MethodDecl->isDeleted(), MethodDecl->isImplicit()});
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
    return llvm::any_of(DefinedMembers, [Kind](const auto &Data) {
      return Data.FunctionKind == Kind && !Data.IsImplicit;
    });
  };

  auto HasImplicitDeletedMember = [&](SpecialMemberFunctionKind Kind) {
    return llvm::any_of(DefinedMembers, [Kind](const auto &Data) {
      return Data.FunctionKind == Kind && Data.IsImplicit && Data.IsDeleted;
    });
  };

  auto IsDeleted = [&](SpecialMemberFunctionKind Kind) {
    return llvm::any_of(DefinedMembers, [Kind](const auto &Data) {
      return Data.FunctionKind == Kind && Data.IsDeleted;
    });
  };

  auto RequireMembers = [&](SpecialMemberFunctionKind Kind1,
                            SpecialMemberFunctionKind Kind2) {
    if (AllowImplicitlyDeletedCopyOrMove && HasImplicitDeletedMember(Kind1) &&
        HasImplicitDeletedMember(Kind2))
      return;

    if (!HasMember(Kind1))
      MissingMembers.push_back(Kind1);

    if (!HasMember(Kind2))
      MissingMembers.push_back(Kind2);
  };

  bool RequireThree =
      HasMember(SpecialMemberFunctionKind::NonDefaultDestructor) ||
      (!AllowSoleDefaultDtor &&
       (HasMember(SpecialMemberFunctionKind::Destructor) ||
        HasMember(SpecialMemberFunctionKind::DefaultDestructor))) ||
      HasMember(SpecialMemberFunctionKind::CopyConstructor) ||
      HasMember(SpecialMemberFunctionKind::CopyAssignment) ||
      HasMember(SpecialMemberFunctionKind::MoveConstructor) ||
      HasMember(SpecialMemberFunctionKind::MoveAssignment);

  bool RequireFive = (!AllowMissingMoveFunctions && RequireThree &&
                      getLangOpts().CPlusPlus11) ||
                     HasMember(SpecialMemberFunctionKind::MoveConstructor) ||
                     HasMember(SpecialMemberFunctionKind::MoveAssignment);

  if (RequireThree) {
    if (!HasMember(SpecialMemberFunctionKind::Destructor) &&
        !HasMember(SpecialMemberFunctionKind::DefaultDestructor) &&
        !HasMember(SpecialMemberFunctionKind::NonDefaultDestructor))
      MissingMembers.push_back(SpecialMemberFunctionKind::Destructor);

    RequireMembers(SpecialMemberFunctionKind::CopyConstructor,
                   SpecialMemberFunctionKind::CopyAssignment);
  }

  if (RequireFive &&
      !(AllowMissingMoveFunctionsWhenCopyIsDeleted &&
        (IsDeleted(SpecialMemberFunctionKind::CopyConstructor) &&
         IsDeleted(SpecialMemberFunctionKind::CopyAssignment)))) {
    assert(RequireThree);
    RequireMembers(SpecialMemberFunctionKind::MoveConstructor,
                   SpecialMemberFunctionKind::MoveAssignment);
  }

  if (!MissingMembers.empty()) {
    llvm::SmallVector<SpecialMemberFunctionKind, 5> DefinedMemberKinds;
    for (const auto &Data : DefinedMembers) {
      if (!Data.IsImplicit)
        DefinedMemberKinds.push_back(Data.FunctionKind);
    }
    diag(ID.first, "class '%0' defines %1 but does not define %2")
        << ID.second << cppcoreguidelines::join(DefinedMemberKinds, " and ")
        << cppcoreguidelines::join(MissingMembers, " or ");
  }
}

} // namespace clang::tidy::cppcoreguidelines
