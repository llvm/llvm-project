//===--- DefineInline.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Logger.h"
#include "Selection.h"
#include "SourceCode.h"
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Deduces the FunctionDecl from a selection. Requires either the function body
// or the function decl to be selected. Returns null if none of the above
// criteria is met.
const FunctionDecl *getSelectedFunction(const SelectionTree::Node *SelNode) {
  const ast_type_traits::DynTypedNode &AstNode = SelNode->ASTNode;
  if (const FunctionDecl *FD = AstNode.get<FunctionDecl>())
    return FD;
  if (AstNode.get<CompoundStmt>() &&
      SelNode->Selected == SelectionTree::Complete) {
    if (const SelectionTree::Node *P = SelNode->Parent)
      return P->ASTNode.get<FunctionDecl>();
  }
  return nullptr;
}

// Checks the decls mentioned in Source are visible in the context of Target.
// Achives that by checking declarations occur before target location in
// translation unit or declared in the same class.
bool checkDeclsAreVisible(const llvm::DenseSet<const Decl *> &DeclRefs,
                          const FunctionDecl *Target, const SourceManager &SM) {
  SourceLocation TargetLoc = Target->getLocation();
  // To be used in visibility check below, decls in a class are visible
  // independent of order.
  const RecordDecl *Class = nullptr;
  if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(Target))
    Class = MD->getParent();

  for (const auto *DR : DeclRefs) {
    // Use canonical decl, since having one decl before target is enough.
    const Decl *D = DR->getCanonicalDecl();
    if (D == Target)
      continue;
    SourceLocation DeclLoc = D->getLocation();

    // FIXME: Allow declarations from different files with include insertion.
    if (!SM.isWrittenInSameFile(DeclLoc, TargetLoc))
      return false;

    // If declaration is before target, then it is visible.
    if (SM.isBeforeInTranslationUnit(DeclLoc, TargetLoc))
      continue;

    // Otherwise they need to be in same class
    if (!Class)
      return false;
    const RecordDecl *Parent = nullptr;
    if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(D))
      Parent = MD->getParent();
    else if (const auto *FD = llvm::dyn_cast<FieldDecl>(D))
      Parent = FD->getParent();
    if (Parent != Class)
      return false;
  }
  return true;
}

// Returns the canonical declaration for the given FunctionDecl. This will
// usually be the first declaration in current translation unit with the
// exception of template specialization.
// For those we return first declaration different than the canonical one.
// Because canonical declaration points to template decl instead of
// specialization.
const FunctionDecl *findTarget(const FunctionDecl *FD) {
  auto CanonDecl = FD->getCanonicalDecl();
  if (!FD->isFunctionTemplateSpecialization())
    return CanonDecl;
  // For specializations CanonicalDecl is the TemplatedDecl, which is not the
  // target we want to inline into. Instead we traverse previous decls to find
  // the first forward decl for this specialization.
  auto PrevDecl = FD;
  while (PrevDecl->getPreviousDecl() != CanonDecl) {
    PrevDecl = PrevDecl->getPreviousDecl();
    assert(PrevDecl && "Found specialization without template decl");
  }
  return PrevDecl;
}

/// Moves definition of a function/method to its declaration location.
/// Before:
/// a.h:
///   void foo();
///
/// a.cc:
///   void foo() { return; }
///
/// ------------------------
/// After:
/// a.h:
///   void foo() { return; }
///
/// a.cc:
///
class DefineInline : public Tweak {
public:
  const char *id() const override final;

  Intent intent() const override { return Intent::Refactor; }
  std::string title() const override {
    return "Move function body to declaration";
  }
  bool hidden() const override { return true; }

  // Returns true when selection is on a function definition that does not
  // make use of any internal symbols.
  bool prepare(const Selection &Sel) override {
    const SelectionTree::Node *SelNode = Sel.ASTSelection.commonAncestor();
    if (!SelNode)
      return false;
    Source = getSelectedFunction(SelNode);
    if (!Source || !Source->isThisDeclarationADefinition())
      return false;
    // Only the last level of template parameter locations are not kept in AST,
    // so if we are inlining a method that is in a templated class, there is no
    // way to verify template parameter names. Therefore we bail out.
    if (auto *MD = llvm::dyn_cast<CXXMethodDecl>(Source)) {
      if (MD->getParent()->isTemplated())
        return false;
    }

    Target = findTarget(Source);
    if (Target == Source) {
      // The only declaration is Source. No other declaration to move function
      // body.
      // FIXME: If we are in an implementation file, figure out a suitable
      // location to put declaration. Possibly using other declarations in the
      // AST.
      return false;
    }

    // Check if the decls referenced in function body are visible in the
    // declaration location.
    if (!checkDeclsAreVisible(getNonLocalDeclRefs(Sel.AST, Source), Target,
                              Sel.AST.getSourceManager()))
      return false;

    return true;
  }

  Expected<Effect> apply(const Selection &Sel) override {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Not implemented yet");
  }

private:
  const FunctionDecl *Source = nullptr;
  const FunctionDecl *Target = nullptr;
};

REGISTER_TWEAK(DefineInline);

} // namespace
} // namespace clangd
} // namespace clang
