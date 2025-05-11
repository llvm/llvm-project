//===--- AddPureVirtualOverride.cpp ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "refactor/Tweak.h"

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
#include <vector>

namespace clang {
namespace clangd {
namespace {

class OverridePureVirtuals : public Tweak {
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
  llvm::MapVector<AccessSpecifier, std::vector<const CXXMethodDecl *>>
      MissingMethodsByAccess;
  // Stores the source locations of existing access specifiers in CurrentDecl.
  llvm::MapVector<AccessSpecifier, SourceLocation> AccessSpecifierLocations;

  // Helper function to gather information before applying the tweak.
  void collectMissingPureVirtuals(const Selection &Sel);
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
    return true; // Continue iterating through all bases
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
  if (!CurrentDecl || !CurrentDecl->getDefinition())
    return false;

  return llvm::any_of(
      CurrentDecl->getDefinition()->bases(), [](CXXBaseSpecifier BaseSpec) {
        const auto *D = BaseSpec.getType()->getAsCXXRecordDecl();
        const auto *Def = D ? D->getDefinition() : nullptr;
        return Def && Def->isAbstract();
      });
}

// Check if the current class has any pure virtual method to be implemented.
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

  // Only offer for polymorphic classes with abstract bases.
  return CurrentDeclDef->isPolymorphic() &&
         hasAbstractBaseAncestor(CurrentDeclDef);
}

// Collects all pure virtual methods that are missing an override in
// CurrentDecl, grouped by their original access specifier.
void OverridePureVirtuals::collectMissingPureVirtuals(const Selection &Sel) {
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

// Free function to generate the string for a group of method overrides.
std::string
generateOverridesStringForGroup(std::vector<const CXXMethodDecl *> Methods,
                                const LangOptions &LangOpts) {
  const auto GetParamString = [&LangOpts](const ParmVarDecl *P) {
    std::string TypeStr = P->getType().getAsString(LangOpts);
    // Unnamed parameter.
    if (P->getNameAsString().empty())
      return TypeStr;

    return llvm::formatv("{0} {1}", std::move(TypeStr), P->getNameAsString())
        .str();
  };

  std::string MethodsString;
  for (const auto *Method : Methods) {
    llvm::SmallVector<std::string> ParamsAsString;
    ParamsAsString.reserve(Method->parameters().size());
    llvm::transform(Method->parameters(), std::back_inserter(ParamsAsString),
                    GetParamString);
    auto Params = llvm::join(ParamsAsString, ", ");

    MethodsString +=
        llvm::formatv(
            "  {0} {1}({2}) {3}override {{\n"
            "    // TODO: Implement this pure virtual method\n"
            "    static_assert(false, \"Method `{1}` is not implemented.\");\n"
            "  }\n",
            Method->getReturnType().getAsString(LangOpts),
            Method->getNameAsString(), std::move(Params),
            std::string(Method->isConst() ? "const " : ""))
            .str();
  }
  return MethodsString;
}

Expected<Tweak::Effect> OverridePureVirtuals::apply(const Selection &Sel) {
  // The correctness of this tweak heavily relies on the accurate population of
  // these members.
  collectMissingPureVirtuals(Sel);

  if (MissingMethodsByAccess.empty()) {
    return llvm::make_error<llvm::StringError>(
        "No pure virtual methods to override.", llvm::inconvertibleErrorCode());
  }

  const auto &SM = Sel.AST->getSourceManager();
  const auto &LangOpts = Sel.AST->getLangOpts();

  tooling::Replacements EditReplacements;
  // Stores text for new access specifier sections // that are not already
  // present in the class.
  // Example:
  //  public:    // ...
  //  protected: // ...
  std::string NewSectionsToAppendText;
  // Tracks if we are adding the first new access section
  // to NewSectionsToAppendText, to manage preceding newlines.
  bool IsFirstNewSection = true;

  // Define the order in which access specifiers should be processed and
  // potentially added.
  constexpr auto AccessOrder = std::array{
      AccessSpecifier::AS_public,
      AccessSpecifier::AS_protected,
      AccessSpecifier::AS_private,
  };

  for (AccessSpecifier AS : AccessOrder) {
    auto *GroupIter = MissingMethodsByAccess.find(AS);
    // Check if there are any missing methods for the current access specifier.
    // No methods to override for this access specifier.
    if (GroupIter == MissingMethodsByAccess.end() || GroupIter->second.empty())
      continue;

    std::string MethodsGroupString =
        generateOverridesStringForGroup(GroupIter->second, LangOpts);

    auto *ExistingSpecLocIter = AccessSpecifierLocations.find(AS);
    if (ExistingSpecLocIter != AccessSpecifierLocations.end()) {
      // Access specifier section already exists in the class.
      // Get location immediately *after* the colon.
      SourceLocation InsertLoc =
          ExistingSpecLocIter->second.getLocWithOffset(1);

      // Create a replacement to insert the method declarations.
      // The replacement is at InsertLoc, has length 0 (insertion), and uses
      // InsertionText.
      std::string InsertionText = "\n" + MethodsGroupString;
      tooling::Replacement Rep(SM, InsertLoc, 0, InsertionText);
      if (auto Err = EditReplacements.add(Rep))
        return llvm::Expected<Tweak::Effect>(std::move(Err));
    } else {
      // Access specifier section does not exist in the class.
      // These methods will be grouped into NewSectionsToAppendText and added
      // towards the end of the class definition.
      if (!IsFirstNewSection)
        NewSectionsToAppendText += "\n";

      NewSectionsToAppendText +=
          getAccessSpelling(AS).str() + ":\n" + MethodsGroupString;
      IsFirstNewSection = false;
    }
  }

  // After processing all access specifiers, add any newly created sections
  // (stored in NewSectionsToAppendText) to the end of the class.
  if (!NewSectionsToAppendText.empty()) {
    // AppendLoc is the SourceLocation of the closing brace '}' of the class.
    // The replacement will insert text *before* this closing brace.
    SourceLocation AppendLoc = CurrentDeclDef->getBraceRange().getEnd();
    std::string FinalAppendText = NewSectionsToAppendText;

    if (!CurrentDeclDef->decls_empty() || !EditReplacements.empty()) {
      FinalAppendText = "\n" + FinalAppendText;
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
