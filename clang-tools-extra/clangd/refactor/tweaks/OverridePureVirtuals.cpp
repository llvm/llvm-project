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
#include <algorithm>
#include <functional>
#include <map>
#include <set>
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
    return CodeAction::REFACTOR_KIND;
  }

private:
  // Stores the CXXRecordDecl of the class being modified.
  const CXXRecordDecl *CurrentDecl = nullptr;
  // Stores pure virtual methods that need overriding, grouped by their original
  // access specifier.
  std::map<AccessSpecifier, std::vector<const CXXMethodDecl *>>
      MissingMethodsByAccess;
  // Stores the source locations of existing access specifiers in CurrentDecl.
  std::map<AccessSpecifier, SourceLocation> AccessSpecifierLocations;

  // Helper function to gather information before applying the tweak.
  void collectMissingPureVirtuals(const Selection &Sel);
};

REGISTER_TWEAK(OverridePureVirtuals)

// Collects all unique, canonical pure virtual methods from a class and its
// entire inheritance hierarchy. This function aims to find methods that *could*
// make a derived class abstract if not implemented.
std::vector<const CXXMethodDecl *>
getAllUniquePureVirtualsFromHierarchy(const CXXRecordDecl *Decl) {
  std::vector<const CXXMethodDecl *> Result;
  llvm::DenseSet<const CXXMethodDecl *> VisitedCanonicalMethods;
  // We declare it as a std::function because we are going to call it
  // recursively.
  std::function<void(const CXXRecordDecl *)> Collect;

  Collect = [&](const CXXRecordDecl *CurrentClass) {
    if (!CurrentClass) {
      return;
    }
    const CXXRecordDecl *Def = CurrentClass->getDefinition();
    if (!Def) {
      return;
    }

    for (const CXXMethodDecl *M : Def->methods()) {
      // Add if its canonical declaration hasn't been processed yet.
      // This ensures each distinct pure virtual method signature is collected
      // once.
      if (M->isPureVirtual() &&
          VisitedCanonicalMethods.insert(M->getCanonicalDecl()).second) {
        Result.emplace_back(M); // Store the specific declaration encountered.
      }
    }

    for (const auto &BaseSpec : Def->bases()) {
      if (const CXXRecordDecl *BaseDef =
              BaseSpec.getType()->getAsCXXRecordDecl()) {
        Collect(BaseDef); // Recursively collect from base classes.
      }
    }
  };

  Collect(Decl);
  return Result;
}

// Gets canonical declarations of methods already overridden or implemented in
// class D.
std::set<const CXXMethodDecl *>
getImplementedOrOverriddenCanonicals(const CXXRecordDecl *D) {
  std::set<const CXXMethodDecl *> ImplementedSet;
  for (const CXXMethodDecl *M : D->methods()) {
    // If M provides an implementation for any virtual method it overrides.
    // A method is an "implementation" if it's virtual and not pure.
    // Or if it directly overrides a base method.
    for (const CXXMethodDecl *OverriddenM : M->overridden_methods()) {
      ImplementedSet.insert(OverriddenM->getCanonicalDecl());
    }
  }
  return ImplementedSet;
}

// Get the location of every colon of the `AccessSpecifier`.
std::map<AccessSpecifier, SourceLocation>
getSpecifierLocations(const CXXRecordDecl *D) {
  std::map<AccessSpecifier, SourceLocation> Locs;
  for (auto *DeclNode : D->decls()) { // Changed to DeclNode to avoid ambiguity
    if (const auto *ASD = llvm::dyn_cast<AccessSpecDecl>(DeclNode)) {
      Locs[ASD->getAccess()] = ASD->getColonLoc();
    }
  }
  return Locs;
}

// Check if the current class has any pure virtual method to be implemented.
bool OverridePureVirtuals::prepare(const Selection &Sel) {
  const SelectionTree::Node *Node = Sel.ASTSelection.commonAncestor();
  if (!Node) {
    return false;
  }

  // Make sure we have a definition.
  CurrentDecl = Node->ASTNode.get<CXXRecordDecl>();
  if (!CurrentDecl || !CurrentDecl->getDefinition()) {
    return false;
  }

  // A class needs overrides if it's abstract itself, or derives from abstract
  // bases and hasn't implemented all necessary methods. A simpler check: if it
  // has any base that is abstract.
  bool HasAbstractBase = false;
  for (const auto &Base : CurrentDecl->bases()) {
    if (const CXXRecordDecl *BaseDecl = Base.getType()->getAsCXXRecordDecl()) {
      if (BaseDecl->getDefinition() &&
          BaseDecl->getDefinition()->isAbstract()) {
        HasAbstractBase = true;
        break;
      }
    }
  }

  // Only offer for polymorphic classes with abstract bases.
  return CurrentDecl->isPolymorphic() && HasAbstractBase;
}

// Collects all pure virtual methods that are missing an override in
// CurrentDecl, grouped by their original access specifier.
void OverridePureVirtuals::collectMissingPureVirtuals(const Selection &Sel) {
  if (!CurrentDecl)
    return;
  CurrentDecl = CurrentDecl->getDefinition(); // Work with the definition
  if (!CurrentDecl)
    return;

  AccessSpecifierLocations = getSpecifierLocations(CurrentDecl);
  MissingMethodsByAccess.clear();

  // Get all unique pure virtual methods from the entire base class hierarchy.
  std::vector<const CXXMethodDecl *> AllPureVirtualsInHierarchy;
  llvm::DenseSet<const CXXMethodDecl *> CanonicalPureVirtualsSeen;

  for (const auto &BaseSpec : CurrentDecl->bases()) {
    if (const CXXRecordDecl *BaseRD =
            BaseSpec.getType()->getAsCXXRecordDecl()) {
      const CXXRecordDecl *BaseDef = BaseRD->getDefinition();
      if (!BaseDef)
        continue;

      std::vector<const CXXMethodDecl *> PuresFromBasePath =
          getAllUniquePureVirtualsFromHierarchy(BaseDef);
      for (const CXXMethodDecl *M : PuresFromBasePath) {
        if (CanonicalPureVirtualsSeen.insert(M->getCanonicalDecl()).second) {
          AllPureVirtualsInHierarchy.emplace_back(M);
        }
      }
    }
  }

  // Get methods already implemented or overridden in CurrentDecl.
  const auto ImplementedOrOverriddenSet =
      getImplementedOrOverriddenCanonicals(CurrentDecl);

  // Filter AllPureVirtualsInHierarchy to find those not in
  // ImplementedOrOverriddenSet.
  for (const CXXMethodDecl *BaseMethod : AllPureVirtualsInHierarchy) {
    bool AlreadyHandled =
        ImplementedOrOverriddenSet.count(BaseMethod->getCanonicalDecl()) > 0;

    if (!AlreadyHandled) {
      // This method needs an override.
      // Group it by its access specifier in its defining class.
      MissingMethodsByAccess[BaseMethod->getAccess()].emplace_back(BaseMethod);
    }
  }
}

// Free function to generate the string for a group of method overrides.
std::string
generateOverridesStringForGroup(std::vector<const CXXMethodDecl *> Methods,
                                const LangOptions &LangOpts) {
  const auto GetParamString = [&LangOpts](const ParmVarDecl *P) {
    std::string TypeStr = P->getType().getAsString(LangOpts);
    if (P->getNameAsString().empty()) {
      // Unnamed parameter.
      return TypeStr;
    }
    return llvm::formatv("{0} {1}", TypeStr, P->getNameAsString()).str();
  };

  std::string MethodsString;
  for (const auto *Method : Methods) {
    std::vector<std::string> ParamsAsString;
    ParamsAsString.reserve(Method->parameters().size());
    std::transform(Method->param_begin(), Method->param_end(),
                   std::back_inserter(ParamsAsString), GetParamString);
    const auto Params = llvm::join(ParamsAsString, ", ");

    MethodsString +=
        llvm::formatv(
            "  {0} {1}({2}) {3}override {{\n"
            "    // TODO: Implement this pure virtual method\n"
            "    static_assert(false, \"Method `{1}` is not implemented.\");\n"
            "  }\n",
            Method->getReturnType().getAsString(LangOpts),
            Method->getNameAsString(), Params,
            std::string(Method->isConst() ? "const " : ""))
            .str();
  }
  return MethodsString;
}

// Helper to get the string spelling of an AccessSpecifier.
std::string getAccessSpecifierSpelling(AccessSpecifier AS) {
  switch (AS) {
  case AS_public:
    return "public";
  case AS_protected:
    return "protected";
  case AS_private:
    return "private";
  case AS_none:
    // Should not typically occur for class members.
    return "";
  }
  // Unreachable.
  return "";
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
    auto GroupIter = MissingMethodsByAccess.find(AS);
    // Check if there are any missing methods for the current access specifier.
    if (GroupIter == MissingMethodsByAccess.end() ||
        GroupIter->second.empty()) {
      // No methods to override for this access specifier.
      continue;
    }

    std::string MethodsGroupString =
        generateOverridesStringForGroup(GroupIter->second, LangOpts);

    auto ExistingSpecLocIter = AccessSpecifierLocations.find(AS);
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
      if (auto Err = EditReplacements.add(Rep)) {
        // Handle error if replacement couldn't be added (e.g. overlaps)
        return llvm::Expected<Tweak::Effect>(std::move(Err));
      }
    } else {
      // Access specifier section does not exist in the class.
      // These methods will be grouped into NewSectionsToAppendText and added
      // towards the end of the class definition.
      if (!IsFirstNewSection) {
        NewSectionsToAppendText += "\n";
      }
      NewSectionsToAppendText +=
          getAccessSpecifierSpelling(AS) + ":\n" + MethodsGroupString;
      IsFirstNewSection = false;
    }
  }

  // After processing all access specifiers, add any newly created sections
  // (stored in NewSectionsToAppendText) to the end of the class.
  if (!NewSectionsToAppendText.empty()) {
    // AppendLoc is the SourceLocation of the closing brace '}' of the class.
    // The replacement will insert text *before* this closing brace.
    SourceLocation AppendLoc = CurrentDecl->getBraceRange().getEnd();
    std::string FinalAppendText = NewSectionsToAppendText;

    if (!CurrentDecl->decls_empty() || !EditReplacements.empty()) {
      FinalAppendText = "\n" + FinalAppendText;
    }

    // Create a replacement to append the new sections.
    tooling::Replacement Rep(SM, AppendLoc, 0, FinalAppendText);
    if (auto Err = EditReplacements.add(Rep)) {
      // Handle error if replacement couldn't be added
      return llvm::Expected<Tweak::Effect>(std::move(Err));
    }
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
