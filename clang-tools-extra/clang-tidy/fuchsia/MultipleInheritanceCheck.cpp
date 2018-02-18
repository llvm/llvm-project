//===--- MultipleInheritanceCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MultipleInheritanceCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

namespace {
AST_MATCHER(CXXRecordDecl, hasBases) {
  if (Node.hasDefinition())
    return Node.getNumBases() > 0;
  return false;
}
} // namespace

// Adds a node (by name) to the interface map, if it was not present in the map
// previously.
void MultipleInheritanceCheck::addNodeToInterfaceMap(const CXXRecordDecl *Node,
                                                     bool isInterface) {
  StringRef Name = Node->getIdentifier()->getName();
  InterfaceMap.insert(std::make_pair(Name, isInterface));
}

// Returns "true" if the boolean "isInterface" has been set to the
// interface status of the current Node. Return "false" if the
// interface status for the current node is not yet known.
bool MultipleInheritanceCheck::getInterfaceStatus(const CXXRecordDecl *Node,
                                                  bool &isInterface) const {
  StringRef Name = Node->getIdentifier()->getName();
  llvm::StringMapConstIterator<bool> Pair = InterfaceMap.find(Name);
  if (Pair == InterfaceMap.end())
    return false;
  isInterface = Pair->second;
  return true;
}

bool MultipleInheritanceCheck::isCurrentClassInterface(
    const CXXRecordDecl *Node) const {
  // Interfaces should have no fields.
  if (!Node->field_empty()) return false;

  // Interfaces should have exclusively pure methods.
  return llvm::none_of(Node->methods(), [](const CXXMethodDecl *M) {
    return M->isUserProvided() && !M->isPure() && !M->isStatic();
  });
}

bool MultipleInheritanceCheck::isInterface(const CXXRecordDecl *Node) {
  // Short circuit the lookup if we have analyzed this record before.
  bool PreviousIsInterfaceResult;
  if (getInterfaceStatus(Node, PreviousIsInterfaceResult))
    return PreviousIsInterfaceResult;

  // To be an interface, all base classes must be interfaces as well.
  for (const auto &I : Node->bases()) {
    if (I.isVirtual()) continue;
    const auto *Ty = I.getType()->getAs<RecordType>();
    if (!Ty) continue;
    const RecordDecl *D = Ty->getDecl()->getDefinition();
    if (!D) continue;
    const auto *Base = cast<CXXRecordDecl>(D);
    if (!isInterface(Base)) {
      addNodeToInterfaceMap(Node, false);
      return false;
    }
  }

  bool CurrentClassIsInterface = isCurrentClassInterface(Node);
  addNodeToInterfaceMap(Node, CurrentClassIsInterface);
  return CurrentClassIsInterface;
}

void MultipleInheritanceCheck::registerMatchers(MatchFinder *Finder) {
  // Requires C++.
  if (!getLangOpts().CPlusPlus)
    return;

  // Match declarations which have bases.
  Finder->addMatcher(cxxRecordDecl(hasBases()).bind("decl"), this);
}

void MultipleInheritanceCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *D = Result.Nodes.getNodeAs<CXXRecordDecl>("decl")) {
    // Check against map to see if if the class inherits from multiple
    // concrete classes
    unsigned NumConcrete = 0;
    for (const auto &I : D->bases()) {
      if (I.isVirtual()) continue;
      const auto *Ty = I.getType()->getAs<RecordType>();
      if (!Ty) continue;
      const auto *Base = cast<CXXRecordDecl>(Ty->getDecl()->getDefinition());
      if (!isInterface(Base)) NumConcrete++;
    }
    
    // Check virtual bases to see if there is more than one concrete 
    // non-virtual base.
    for (const auto &V : D->vbases()) {
      const auto *Ty = V.getType()->getAs<RecordType>();
      if (!Ty) continue;
      const auto *Base = cast<CXXRecordDecl>(Ty->getDecl()->getDefinition());
      if (!isInterface(Base)) NumConcrete++;
    }

    if (NumConcrete > 1) {
      diag(D->getLocStart(),
           "inheriting mulitple classes that aren't "
           "pure virtual is discouraged");
    }
  }
}

}  // namespace fuchsia
}  // namespace tidy
}  // namespace clang
