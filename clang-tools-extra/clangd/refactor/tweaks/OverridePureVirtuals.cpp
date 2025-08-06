//===--- OverridePureVirtuals.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tweak to automatically generate stubs for pure virtual methods inherited from
// base classes.
//
// Purpose:
//   - Simplifies making a derived class concrete by automating the creation of
//     required method overrides from abstract bases.
//
// Tweak Summary:
//
// 1. Activation Conditions (prepare):
//    - The tweak activates when the cursor is over a C++ class definition.
//    - The class must be abstract (it, or its base classes, have unimplemented
//      pure virtual functions).
//    - It must also inherit from at least one other abstract class.
//
// 2. Identifying Missing Methods:
//    - The tweak scans the inheritance hierarchy of the current class.
//    - It identifies all unique pure virtual methods from base classes
//      that are not yet implemented or overridden.
//    - These missing methods are then grouped by their original access
//      specifier (e.g., public, protected).
//
// 3. Code Generation and Insertion:
//    - For each group of missing methods, stubs are inserted.
//    - If an access specifier section (like `public:`) exists, stubs are
//      inserted there; otherwise, a new section is created and appended.
//    - Each generated stub includes the `override` keyword, a `// TODO:`
//      comment, and a `static_assert(false, ...)` to force a compile-time
//      error if the method remains unimplemented.
//    - The base method's signature is adjusted (e.g., `virtual` and `= 0`
//      are removed for the override).
//
// 4. Code Action Provided:
//    - A single code action titled "Override pure virtual methods" is offered.
//    - Applying this action results in a single source file modification
//      containing all the generated method stubs.
//
// Example:
//
//  class Base {
//  public:
//    virtual void publicMethod() = 0;
//  protected:
//    virtual auto privateMethod() const -> int = 0;
//  };
//
// Before:
//                              // cursor here
//  class Derived : public Base {}^;
//
// After:
//
// class Derived : public Base {
//  public:
//   void publicMethod() override {
//     // TODO: Implement this pure virtual method.
//     static_assert(false, "Method `publicMethod` is not implemented.");
//   }
//
//  protected:
//   auto privateMethod() const -> int override {
//     // TODO: Implement this pure virtual method.
//     static_assert(false, "Method `privateMethod` is not implemented.");
//   }
// };
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"
#include "support/Token.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace clang {
namespace clangd {
namespace {

// This function removes the "virtual" and the "= 0" at the end;
// e.g.:
//   "virtual void foo(int var = 0) = 0"  // input.
//   "void foo(int var = 0)"              // output.
std::string removePureVirtualSyntax(const std::string &MethodDecl,
                                    const LangOptions &LangOpts) {
  assert(!MethodDecl.empty());

  TokenStream TS = lex(MethodDecl, LangOpts);

  std::string DeclString;
  for (const clangd::Token &Tk : TS.tokens()) {
    if (Tk.Kind == clang::tok::raw_identifier && Tk.text() == "virtual")
      continue;

    // If the ending two tokens are "= 0", we break here and we already have the
    // method's string without the pure virtual syntax.
    const auto &Next = Tk.next();
    if (Next.next().Kind == tok::eof && Tk.Kind == clang::tok::equal &&
        Next.text() == "0")
      break;

    DeclString += Tk.text();
    if (Tk.Kind != tok::l_paren && Next.Kind != tok::comma &&
        Next.Kind != tok::r_paren && Next.Kind != tok::l_paren)
      DeclString += ' ';
  }
  // Trim the last whitespace.
  if (DeclString.back() == ' ')
    DeclString.pop_back();

  return DeclString;
}

class OverridePureVirtuals final : public Tweak {
public:
  const char *id() const final; // defined by REGISTER_TWEAK.
  bool prepare(const Selection &Sel) override;
  Expected<Effect> apply(const Selection &Sel) override;
  std::string title() const override { return "Override pure virtual methods"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::QUICKFIX_KIND;
  }

private:
  // Stores the CXXRecordDecl of the class being modified.
  const CXXRecordDecl *CurrentDeclDef = nullptr;
  // Stores pure virtual methods that need overriding, grouped by their original
  // access specifier.
  llvm::MapVector<AccessSpecifier, llvm::SmallVector<const CXXMethodDecl *>>
      MissingMethodsByAccess;
  // Stores the source locations of existing access specifiers in CurrentDecl.
  llvm::MapVector<AccessSpecifier, SourceLocation> AccessSpecifierLocations;
  // Helper function to gather information before applying the tweak.
  void collectMissingPureVirtuals();
};

REGISTER_TWEAK(OverridePureVirtuals)

// Function to get all unique pure virtual methods from the entire
// base class hierarchy of CurrentDeclDef.
llvm::SmallVector<const clang::CXXMethodDecl *>
getAllUniquePureVirtualsFromBaseHierarchy(
    const clang::CXXRecordDecl *CurrentDeclDef) {
  llvm::SmallVector<const clang::CXXMethodDecl *> AllPureVirtualsInHierarchy;
  llvm::DenseSet<const clang::CXXMethodDecl *> CanonicalPureVirtualsSeen;

  if (!CurrentDeclDef || !CurrentDeclDef->getDefinition())
    return AllPureVirtualsInHierarchy;

  const clang::CXXRecordDecl *Def = CurrentDeclDef->getDefinition();

  Def->forallBases([&](const clang::CXXRecordDecl *BaseDefinition) {
    for (const clang::CXXMethodDecl *Method : BaseDefinition->methods()) {
      if (Method->isPureVirtual() &&
          CanonicalPureVirtualsSeen.insert(Method->getCanonicalDecl()).second)
        AllPureVirtualsInHierarchy.emplace_back(Method);
    }
    // Continue iterating through all bases.
    return true;
  });

  return AllPureVirtualsInHierarchy;
}

// Gets canonical declarations of methods already overridden or implemented in
// class D.
llvm::SetVector<const CXXMethodDecl *>
getImplementedOrOverriddenCanonicals(const CXXRecordDecl *D) {
  llvm::SetVector<const CXXMethodDecl *> ImplementedSet;
  for (const CXXMethodDecl *M : D->methods()) {
    // If M provides an implementation for any virtual method it overrides.
    // A method is an "implementation" if it's virtual and not pure.
    // Or if it directly overrides a base method.
    for (const CXXMethodDecl *OverriddenM : M->overridden_methods())
      ImplementedSet.insert(OverriddenM->getCanonicalDecl());
  }
  return ImplementedSet;
}

// Get the location of every colon of the `AccessSpecifier`.
llvm::MapVector<AccessSpecifier, SourceLocation>
getSpecifierLocations(const CXXRecordDecl *D) {
  llvm::MapVector<AccessSpecifier, SourceLocation> Locs;
  for (auto *DeclNode : D->decls()) {
    if (const auto *ASD = llvm::dyn_cast<AccessSpecDecl>(DeclNode))
      Locs[ASD->getAccess()] = ASD->getColonLoc();
  }
  return Locs;
}

bool hasAbstractBaseAncestor(const clang::CXXRecordDecl *CurrentDecl) {
  assert(CurrentDecl && CurrentDecl->getDefinition());

  return llvm::any_of(
      CurrentDecl->getDefinition()->bases(), [](CXXBaseSpecifier BaseSpec) {
        const auto *D = BaseSpec.getType()->getAsCXXRecordDecl();
        const auto *Def = D ? D->getDefinition() : nullptr;
        return Def && Def->isAbstract();
      });
}

// The tweak is available if the selection is over an abstract C++ class
// definition that also inherits from at least one other abstract class.
bool OverridePureVirtuals::prepare(const Selection &Sel) {
  const SelectionTree::Node *Node = Sel.ASTSelection.commonAncestor();
  if (!Node)
    return false;

  // Make sure we have a definition.
  CurrentDeclDef = Node->ASTNode.get<CXXRecordDecl>();
  if (!CurrentDeclDef || !CurrentDeclDef->getDefinition())
    return false;

  // From now on, we should work with the definition.
  CurrentDeclDef = CurrentDeclDef->getDefinition();

  // Only offer for abstract classes with abstract bases.
  return CurrentDeclDef->isAbstract() &&
         hasAbstractBaseAncestor(CurrentDeclDef);
}

// Collects all pure virtual methods from base classes that `CurrentDeclDef` has
// not yet overridden, grouped by their original access specifier.
//
// Results are stored in `MissingMethodsByAccess` and `AccessSpecifierLocations`
// is also populated.
void OverridePureVirtuals::collectMissingPureVirtuals() {
  if (!CurrentDeclDef)
    return;

  AccessSpecifierLocations = getSpecifierLocations(CurrentDeclDef);
  MissingMethodsByAccess.clear();

  // Get all unique pure virtual methods from the entire base class hierarchy.
  llvm::SmallVector<const CXXMethodDecl *> AllPureVirtualsInHierarchy =
      getAllUniquePureVirtualsFromBaseHierarchy(CurrentDeclDef);

  // Get methods already implemented or overridden in CurrentDecl.
  const auto ImplementedOrOverriddenSet =
      getImplementedOrOverriddenCanonicals(CurrentDeclDef);

  // Filter AllPureVirtualsInHierarchy to find those not in
  // ImplementedOrOverriddenSet, which needs to be overriden.
  for (const CXXMethodDecl *BaseMethod : AllPureVirtualsInHierarchy) {
    bool AlreadyHandled = ImplementedOrOverriddenSet.contains(BaseMethod);
    if (!AlreadyHandled)
      MissingMethodsByAccess[BaseMethod->getAccess()].emplace_back(BaseMethod);
  }
}

std::string generateOverrideString(const CXXMethodDecl *Method,
                                   const LangOptions &LangOpts) {
  std::string MethodDecl;
  auto OS = llvm::raw_string_ostream(MethodDecl);
  Method->print(OS);

  return llvm::formatv(
             "\n  {0} override {{\n"
             "    // TODO: Implement this pure virtual method.\n"
             "    static_assert(false, \"Method `{1}` is not implemented.\");\n"
             "  }",
             removePureVirtualSyntax(MethodDecl, LangOpts), Method->getName())
      .str();
}

// Free function to generate the string for a group of method overrides.
std::string generateOverridesStringForGroup(
    llvm::SmallVector<const CXXMethodDecl *> Methods,
    const LangOptions &LangOpts) {
  llvm::SmallVector<std::string> MethodsString;
  MethodsString.reserve(Methods.size());

  for (const CXXMethodDecl *Method : Methods) {
    MethodsString.emplace_back(generateOverrideString(Method, LangOpts));
  }

  return llvm::join(MethodsString, "\n") + '\n';
}

Expected<Tweak::Effect> OverridePureVirtuals::apply(const Selection &Sel) {
  // The correctness of this tweak heavily relies on the accurate population of
  // these members.
  collectMissingPureVirtuals();
  // The `prepare` should prevent this. If the prepare identifies an abstract
  // method, then is must have missing methods.
  assert(!MissingMethodsByAccess.empty());

  const auto &SM = Sel.AST->getSourceManager();
  const auto &LangOpts = Sel.AST->getLangOpts();

  tooling::Replacements EditReplacements;
  // Stores text for new access specifier sections that are not already present
  // in the class.
  // Example:
  //  public:    // ...
  //  protected: // ...
  std::string NewSectionsToAppendText;

  for (const auto &[AS, Methods] : MissingMethodsByAccess) {
    assert(!Methods.empty());

    std::string MethodsGroupString =
        generateOverridesStringForGroup(Methods, LangOpts);

    auto *ExistingSpecLocIter = AccessSpecifierLocations.find(AS);
    bool ASExists = ExistingSpecLocIter != AccessSpecifierLocations.end();
    if (ASExists) {
      // Access specifier section already exists in the class.
      // Get location immediately *after* the colon.
      SourceLocation InsertLoc =
          ExistingSpecLocIter->second.getLocWithOffset(1);

      // Create a replacement to insert the method declarations.
      // The replacement is at InsertLoc, has length 0 (insertion), and uses
      // InsertionText.
      std::string InsertionText = MethodsGroupString;
      tooling::Replacement Rep(SM, InsertLoc, 0, InsertionText);
      if (auto Err = EditReplacements.add(Rep))
        return llvm::Expected<Tweak::Effect>(std::move(Err));
    } else {
      // Access specifier section does not exist in the class.
      // These methods will be grouped into NewSectionsToAppendText and added
      // towards the end of the class definition.
      NewSectionsToAppendText +=
          getAccessSpelling(AS).str() + ':' + MethodsGroupString;
    }
  }

  // After processing all access specifiers, add any newly created sections
  // (stored in NewSectionsToAppendText) to the end of the class.
  if (!NewSectionsToAppendText.empty()) {
    // AppendLoc is the SourceLocation of the closing brace '}' of the class.
    // The replacement will insert text *before* this closing brace.
    SourceLocation AppendLoc = CurrentDeclDef->getBraceRange().getEnd();
    std::string FinalAppendText = std::move(NewSectionsToAppendText);

    if (!CurrentDeclDef->decls_empty() || !EditReplacements.empty()) {
      FinalAppendText = '\n' + FinalAppendText;
    }

    // Create a replacement to append the new sections.
    tooling::Replacement Rep(SM, AppendLoc, 0, FinalAppendText);
    if (auto Err = EditReplacements.add(Rep))
      return llvm::Expected<Tweak::Effect>(std::move(Err));
  }

  if (EditReplacements.empty()) {
    return llvm::make_error<llvm::StringError>(
        "No changes to apply (internal error or no methods generated).",
        llvm::inconvertibleErrorCode());
  }

  // Return the collected replacements as the effect of this tweak.
  return Effect::mainFileEdit(SM, EditReplacements);
}

} // namespace
} // namespace clangd
} // namespace clang
