//===--- SymbolUSRFinder.cpp - Clang refactoring library ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements methods that find the set of USRs that correspond to
/// a symbol that's required for a refactoring operation.
///
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Tooling/Refactor/RefactoringActionFinder.h"
#include "clang/Tooling/Refactor/USRFinder.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

using namespace clang;
using namespace clang::tooling::rename;

namespace {

/// \brief NamedDeclFindingConsumer delegates finding USRs of a found Decl to
/// \c AdditionalUSRFinder. \c AdditionalUSRFinder adds USRs of ctors and dtor
/// if the found declaration refers to a class and adds USRs of all overridden
/// methods if the declaration refers to a virtual C++ method or an ObjC method.
class AdditionalUSRFinder : public RecursiveASTVisitor<AdditionalUSRFinder> {
public:
  AdditionalUSRFinder(const Decl *FoundDecl, ASTContext &Context)
      : FoundDecl(FoundDecl), Context(Context) {}

  llvm::StringSet<> Find() {
    llvm::StringSet<> USRSet;

    // Fill OverriddenMethods and PartialSpecs storages.
    TraverseDecl(Context.getTranslationUnitDecl());
    if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(FoundDecl)) {
      addUSRsOfOverridenFunctions(MethodDecl, USRSet);
      // FIXME: Use a more efficient/optimal algorithm to find the related
      // methods.
      for (const auto &OverriddenMethod : OverriddenMethods) {
        if (checkIfOverriddenFunctionAscends(OverriddenMethod, USRSet))
          USRSet.insert(getUSRForDecl(OverriddenMethod));
      }
    } else if (const auto *RecordDecl = dyn_cast<CXXRecordDecl>(FoundDecl)) {
      handleCXXRecordDecl(RecordDecl, USRSet);
    } else if (const auto *TemplateDecl =
                   dyn_cast<ClassTemplateDecl>(FoundDecl)) {
      handleClassTemplateDecl(TemplateDecl, USRSet);
    } else if (const auto *MethodDecl = dyn_cast<ObjCMethodDecl>(FoundDecl)) {
      addUSRsOfOverriddenObjCMethods(MethodDecl, USRSet);
      for (const auto &PotentialOverrider : PotentialObjCMethodOverridders)
        if (checkIfPotentialObjCMethodOverriddes(PotentialOverrider, USRSet))
          USRSet.insert(getUSRForDecl(PotentialOverrider));
    } else {
      USRSet.insert(getUSRForDecl(FoundDecl));
    }
    return USRSet;
  }

  bool VisitCXXMethodDecl(const CXXMethodDecl *MethodDecl) {
    if (MethodDecl->isVirtual())
      OverriddenMethods.push_back(MethodDecl);
    return true;
  }

  bool VisitObjCMethodDecl(const ObjCMethodDecl *MethodDecl) {
    if (const auto *FoundMethodDecl = dyn_cast<ObjCMethodDecl>(FoundDecl))
      if (DeclarationName::compare(MethodDecl->getDeclName(),
                                   FoundMethodDecl->getDeclName()) == 0 &&
          MethodDecl->isOverriding())
        PotentialObjCMethodOverridders.push_back(MethodDecl);
    return true;
  }

  bool VisitClassTemplatePartialSpecializationDecl(
      const ClassTemplatePartialSpecializationDecl *PartialSpec) {
    if (!isa<ClassTemplateDecl>(FoundDecl) && !isa<CXXRecordDecl>(FoundDecl))
      return true;
    PartialSpecs.push_back(PartialSpec);
    return true;
  }

private:
  void handleCXXRecordDecl(const CXXRecordDecl *RecordDecl,
                           llvm::StringSet<> &USRSet) {
    const auto *RD = RecordDecl->getDefinition();
    if (!RD) {
      USRSet.insert(getUSRForDecl(RecordDecl));
      return;
    }
    if (const auto *ClassTemplateSpecDecl =
            dyn_cast<ClassTemplateSpecializationDecl>(RD))
      handleClassTemplateDecl(ClassTemplateSpecDecl->getSpecializedTemplate(),
                              USRSet);
    addUSRsOfCtorDtors(RD, USRSet);
  }

  void handleClassTemplateDecl(const ClassTemplateDecl *TemplateDecl,
                               llvm::StringSet<> &USRSet) {
    for (const auto *Specialization : TemplateDecl->specializations())
      addUSRsOfCtorDtors(Specialization, USRSet);

    for (const auto *PartialSpec : PartialSpecs) {
      if (PartialSpec->getSpecializedTemplate() == TemplateDecl)
        addUSRsOfCtorDtors(PartialSpec, USRSet);
    }
    addUSRsOfCtorDtors(TemplateDecl->getTemplatedDecl(), USRSet);
  }

  void addUSRsOfCtorDtors(const CXXRecordDecl *RecordDecl,
                          llvm::StringSet<> &USRSet) {
    const CXXRecordDecl *RD = RecordDecl;
    RecordDecl = RD->getDefinition();
    if (!RecordDecl) {
      USRSet.insert(getUSRForDecl(RD));
      return;
    }

    for (const auto *CtorDecl : RecordDecl->ctors()) {
      auto USR = getUSRForDecl(CtorDecl);
      if (!USR.empty())
        USRSet.insert(USR);
    }

    auto USR = getUSRForDecl(RecordDecl->getDestructor());
    if (!USR.empty())
      USRSet.insert(USR);
    USRSet.insert(getUSRForDecl(RecordDecl));
  }

  void addUSRsOfOverridenFunctions(const CXXMethodDecl *MethodDecl,
                                   llvm::StringSet<> &USRSet) {
    USRSet.insert(getUSRForDecl(MethodDecl));
    // Recursively visit each OverridenMethod.
    for (const auto &OverriddenMethod : MethodDecl->overridden_methods())
      addUSRsOfOverridenFunctions(OverriddenMethod, USRSet);
  }

  bool checkIfOverriddenFunctionAscends(const CXXMethodDecl *MethodDecl,
                                        const llvm::StringSet<> &USRSet) {
    for (const auto &OverriddenMethod : MethodDecl->overridden_methods()) {
      if (USRSet.find(getUSRForDecl(OverriddenMethod)) != USRSet.end())
        return true;
      return checkIfOverriddenFunctionAscends(OverriddenMethod, USRSet);
    }
    return false;
  }

  /// \brief Recursively visit all the methods which the given method
  /// declaration overrides and adds them to the USR set.
  void addUSRsOfOverriddenObjCMethods(const ObjCMethodDecl *MethodDecl,
                                      llvm::StringSet<> &USRSet) {
    // Exit early if this method was already visited.
    if (!USRSet.insert(getUSRForDecl(MethodDecl)).second)
      return;
    SmallVector<const ObjCMethodDecl *, 8> Overrides;
    MethodDecl->getOverriddenMethods(Overrides);
    for (const auto &OverriddenMethod : Overrides)
      addUSRsOfOverriddenObjCMethods(OverriddenMethod, USRSet);
  }

  /// \brief Returns true if the given Objective-C method overrides the
  /// found Objective-C method declaration.
  bool checkIfPotentialObjCMethodOverriddes(const ObjCMethodDecl *MethodDecl,
                                            const llvm::StringSet<> &USRSet) {
    SmallVector<const ObjCMethodDecl *, 8> Overrides;
    MethodDecl->getOverriddenMethods(Overrides);
    for (const auto &OverriddenMethod : Overrides) {
      if (USRSet.find(getUSRForDecl(OverriddenMethod)) != USRSet.end())
        return true;
      if (checkIfPotentialObjCMethodOverriddes(OverriddenMethod, USRSet))
        return true;
    }
    return false;
  }

  const Decl *FoundDecl;
  ASTContext &Context;
  std::vector<const CXXMethodDecl *> OverriddenMethods;
  std::vector<const ClassTemplatePartialSpecializationDecl *> PartialSpecs;
  /// \brief An array of Objective-C methods that potentially override the
  /// found Objective-C method declaration \p FoundDecl.
  std::vector<const ObjCMethodDecl *> PotentialObjCMethodOverridders;
};
} // end anonymous namespace

namespace clang {
namespace tooling {

llvm::StringSet<> findSymbolsUSRSet(const NamedDecl *FoundDecl,
                                    ASTContext &Context) {
  return AdditionalUSRFinder(FoundDecl, Context).Find();
}

} // end namespace tooling
} // end namespace clang
