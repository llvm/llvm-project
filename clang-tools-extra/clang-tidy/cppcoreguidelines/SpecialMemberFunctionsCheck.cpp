//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecialMemberFunctionsCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {

enum SpecialMemberFunctions : uint8_t {
  None = 0,
  Dtor = 1 << 0,
  DefaultDtor = 1 << 1,
  NonDefaultDtor = 1 << 2,
  CopyCtor = 1 << 3,
  CopyAssignment = 1 << 4,
  CopyOps = CopyCtor | CopyAssignment,
  MoveCtor = 1 << 5,
  MoveAssignment = 1 << 6,
  MoveOps = MoveCtor | MoveAssignment,
  LLVM_MARK_AS_BITMASK_ENUM(MoveAssignment),
};

} // namespace

static StringRef toString(size_t K) {
  static constexpr StringRef EnumToStringMap[] = {
      "a destructor",
      "a default destructor",
      "a non-default destructor",
      "a copy constructor",
      "a copy assignment operator",
      "a move constructor",
      "a move assignment operator",
  };
  return EnumToStringMap[K];
}

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

static std::string joinSMFs(SpecialMemberFunctions SMFs, StringRef AndOr) {
  assert(SMFs && "List of defined or undefined members should never be empty.");
  std::string Buffer;
  const size_t TotalSMFs = llvm::popcount(llvm::to_underlying(SMFs));
  for (size_t SMFsLeft = TotalSMFs, I = 0; SMFsLeft > 0; ++I) {
    if (!(SMFs & (1 << I)))
      continue;
    if (SMFsLeft != TotalSMFs)
      Buffer += SMFsLeft == 1 ? AndOr : ", ";
    Buffer += toString(I);
    --SMFsLeft;
  }
  return Buffer;
}

void SpecialMemberFunctionsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxRecordDecl().bind("decl"), this);
}

void SpecialMemberFunctionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto &Class = *Result.Nodes.getNodeAs<CXXRecordDecl>("decl");
  if (IgnoreMacros && Class.getBeginLoc().isMacroID() &&
      Class.getEndLoc().isMacroID())
    return;

  SpecialMemberFunctions DefinedSMFs{};
  SpecialMemberFunctions ImplicitSMFs{};
  SpecialMemberFunctions DeletedSMFs{};
  for (const CXXMethodDecl *Method : Class.methods()) {
    SpecialMemberFunctions NewSMF{};
    if (Method->isCopyAssignmentOperator()) {
      NewSMF = CopyAssignment;
    } else if (Method->isMoveAssignmentOperator()) {
      NewSMF = MoveAssignment;
    } else if (const auto *Destructor = dyn_cast<CXXDestructorDecl>(Method)) {
      if (!Destructor->isDefined())
        NewSMF = Dtor;
      else if (Destructor->getDefinition()->isDefaulted())
        NewSMF = DefaultDtor;
      else
        NewSMF = NonDefaultDtor;
    } else if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(Method)) {
      if (Constructor->isCopyConstructor())
        NewSMF = CopyCtor;
      else if (Constructor->isMoveConstructor())
        NewSMF = MoveCtor;
    }

    if (Method->isImplicit())
      ImplicitSMFs |= NewSMF;
    else
      DefinedSMFs |= NewSMF;
    if (Method->isDeleted())
      DeletedSMFs |= NewSMF;
  }

  if (!DefinedSMFs)
    return; // Class follows rule of 0.

  if (AllowSoleDefaultDtor && !(DefinedSMFs & ~(Dtor | DefaultDtor)))
    return;

  SpecialMemberFunctions RequiredSMFs{};
  if (!(AllowImplicitlyDeletedCopyOrMove &&
        (ImplicitSMFs & DeletedSMFs & CopyOps) == CopyOps))
    RequiredSMFs |= CopyOps;

  if (!(DefinedSMFs & (Dtor | NonDefaultDtor | DefaultDtor)))
    RequiredSMFs |= Dtor;

  if (getLangOpts().CPlusPlus11 &&
      ((DefinedSMFs & MoveOps) || !AllowMissingMoveFunctions) &&
      !(AllowImplicitlyDeletedCopyOrMove &&
        (ImplicitSMFs & DeletedSMFs & MoveOps) == MoveOps) &&
      !(AllowMissingMoveFunctionsWhenCopyIsDeleted &&
        (DeletedSMFs & CopyOps) == CopyOps))
    RequiredSMFs |= MoveOps;

  const SpecialMemberFunctions MissingSMFs = RequiredSMFs & ~DefinedSMFs;
  if (!MissingSMFs)
    return;

  diag(Class.getLocation(), "class %0 defines %1 but does not define %2")
      << &Class << joinSMFs(DefinedSMFs, " and ")
      << joinSMFs(MissingSMFs, " or ");
}

} // namespace clang::tidy::cppcoreguidelines
