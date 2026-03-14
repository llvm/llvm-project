//===--- SpecialMembers.cpp - Generate C++ special member functions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ParsedAST.h"
#include "refactor/InsertionPoint.h"
#include "refactor/Tweak.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {

// Returns code to declare missing copy/move constructors/assignment operators.
// They will be deleted or defaulted to match the class's current state.
std::string buildSpecialMemberDeclarations(const CXXRecordDecl &Class) {
  struct Members {
    const CXXMethodDecl *Copy = nullptr;
    const CXXMethodDecl *Move = nullptr;
  } Ctor, Assign;

  for (const auto &M : Class.methods()) {
    if (M->isCopyAssignmentOperator())
      Assign.Copy = M;
    else if (M->isMoveAssignmentOperator())
      Assign.Move = M;
    if (const auto *C = llvm::dyn_cast<CXXConstructorDecl>(M)) {
      if (C->isCopyConstructor())
        Ctor.Copy = C;
      else if (C->isMoveConstructor())
        Ctor.Move = C;
    }
  }

  std::string S;
  llvm::raw_string_ostream OS(S);

  auto PrintMember = [&](const CXXMethodDecl *D, const char *MemberPattern,
                         const char *ParmPattern) {
    if (D && !D->isImplicit())
      return;
    bool Delete = !D || D->isDeleted();
    OS << llvm::formatv(
        "{0} = {1};\n",
        llvm::formatv(MemberPattern, Class.getName(),
                      llvm::formatv(ParmPattern, Class.getName())),
        Delete ? "delete" : "default");
  };
  auto PrintMembers = [&](const Members &M, const char *MemberPattern) {
    PrintMember(M.Copy, MemberPattern, /*ParmPattern=*/"const {0}&");
    PrintMember(M.Move, MemberPattern, /*ParmPattern=*/"{0}&&");
  };
  PrintMembers(Ctor, /*MemberPattern=*/"{0}({1})");
  PrintMembers(Assign, /*MemberPattern=*/"{0} &operator=({1})");

  return S;
}

// A tweak that adds missing declarations of copy & move constructors.
//
// e.g. given `struct ^S{};`, produces:
//   struct S {
//     S(const S&) = default;
//     S(S&&) = default;
//     S &operator=(const S&) = default;
//     S &operator=(S&&) = default;
//   };
//
// Added members are defaulted or deleted to approximately preserve semantics.
// (May not be a strict no-op when they were not implicitly declared).
//
// Having these spelled out is useful:
//  - to understand the implicit behavior
//  - to avoid relying on the implicit behavior
//  - as a baseline for explicit modification
class SpecialMembers : public Tweak {
public:
  const char *id() const final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  std::string title() const override {
    return llvm::formatv("Declare implicit {0} members",
                         NeedCopy ? NeedMove ? "copy/move" : "copy" : "move");
  }

  bool prepare(const Selection &Inputs) override {
    // This tweak relies on =default and =delete.
    if (!Inputs.AST->getLangOpts().CPlusPlus11)
      return false;

    // Trigger only on class definitions.
    if (auto *N = Inputs.ASTSelection.commonAncestor())
      Class = const_cast<CXXRecordDecl *>(N->ASTNode.get<CXXRecordDecl>());
    if (!Class || !Class->isThisDeclarationADefinition() || Class->isUnion())
      return false;

    // Tweak is only available if some members are missing.
    NeedCopy = !Class->hasUserDeclaredCopyConstructor() ||
               !Class->hasUserDeclaredCopyAssignment();
    NeedMove = !Class->hasUserDeclaredMoveAssignment() ||
               !Class->hasUserDeclaredMoveConstructor();
    return NeedCopy || NeedMove;
  }

  Expected<Effect> apply(const Selection &Inputs) override {
    // Implicit special members are created lazily by clang.
    // We need them so we can tell whether they should be =default or =delete.
    Inputs.AST->getSema().ForceDeclarationOfImplicitMembers(Class);
    std::string Code = buildSpecialMemberDeclarations(*Class);

    // Prefer to place the new members...
    std::vector<Anchor> Anchors = {
        // Below the default constructor
        {[](const Decl *D) {
           if (const auto *CCD = llvm::dyn_cast<CXXConstructorDecl>(D))
             return CCD->isDefaultConstructor();
           return false;
         },
         Anchor::Below},
        // Above existing constructors
        {[](const Decl *D) { return llvm::isa<CXXConstructorDecl>(D); },
         Anchor::Above},
        // At the top of the public section
        {[](const Decl *D) { return true; }, Anchor::Above},
    };
    auto Edit = insertDecl(Code, *Class, std::move(Anchors), AS_public);
    if (!Edit)
      return Edit.takeError();
    return Effect::mainFileEdit(Inputs.AST->getSourceManager(),
                                tooling::Replacements{std::move(*Edit)});
  }

private:
  bool NeedCopy = false, NeedMove = false;
  CXXRecordDecl *Class = nullptr;
};
REGISTER_TWEAK(SpecialMembers)

} // namespace
} // namespace clangd
} // namespace clang
