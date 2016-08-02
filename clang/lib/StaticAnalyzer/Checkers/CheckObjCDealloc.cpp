//==- CheckObjCDealloc.cpp - Check ObjC -dealloc implementation --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a CheckObjCDealloc, a checker that
//  analyzes an Objective-C class's implementation to determine if it
//  correctly implements -dealloc.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/LangOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

static bool scan_ivar_release(Stmt *S, ObjCIvarDecl *ID,
                              const ObjCPropertyDecl *PD,
                              Selector Release,
                              IdentifierInfo* SelfII,
                              ASTContext &Ctx) {

  // [mMyIvar release]
  if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getSelector() == Release)
      if (ME->getInstanceReceiver())
        if (Expr *Receiver = ME->getInstanceReceiver()->IgnoreParenCasts())
          if (ObjCIvarRefExpr *E = dyn_cast<ObjCIvarRefExpr>(Receiver))
            if (E->getDecl() == ID)
              return true;

  // [self setMyIvar:nil];
  if (ObjCMessageExpr *ME = dyn_cast<ObjCMessageExpr>(S))
    if (ME->getInstanceReceiver())
      if (Expr *Receiver = ME->getInstanceReceiver()->IgnoreParenCasts())
        if (DeclRefExpr *E = dyn_cast<DeclRefExpr>(Receiver))
          if (E->getDecl()->getIdentifier() == SelfII)
            if (ME->getMethodDecl() == PD->getSetterMethodDecl() &&
                ME->getNumArgs() == 1 &&
                ME->getArg(0)->isNullPointerConstant(Ctx,
                                              Expr::NPC_ValueDependentIsNull))
              return true;

  // self.myIvar = nil;
  if (BinaryOperator* BO = dyn_cast<BinaryOperator>(S))
    if (BO->isAssignmentOp())
      if (ObjCPropertyRefExpr *PRE =
           dyn_cast<ObjCPropertyRefExpr>(BO->getLHS()->IgnoreParenCasts()))
        if (PRE->isExplicitProperty() && PRE->getExplicitProperty() == PD)
            if (BO->getRHS()->isNullPointerConstant(Ctx,
                                            Expr::NPC_ValueDependentIsNull)) {
              // This is only a 'release' if the property kind is not
              // 'assign'.
              return PD->getSetterKind() != ObjCPropertyDecl::Assign;
            }

  // Recurse to children.
  for (Stmt *SubStmt : S->children())
    if (SubStmt && scan_ivar_release(SubStmt, ID, PD, Release, SelfII, Ctx))
      return true;

  return false;
}

static void checkObjCDealloc(const CheckerBase *Checker,
                             const ObjCImplementationDecl *D,
                             const LangOptions &LOpts, BugReporter &BR) {

  assert (LOpts.getGC() != LangOptions::GCOnly);

  ASTContext &Ctx = BR.getContext();
  const ObjCInterfaceDecl *ID = D->getClassInterface();

  // Does the class contain any ivars that are pointers (or id<...>)?
  // If not, skip the check entirely.
  // NOTE: This is motivated by PR 2517:
  //        http://llvm.org/bugs/show_bug.cgi?id=2517

  bool containsPointerIvar = false;

  for (const auto *Ivar : ID->ivars()) {
    QualType T = Ivar->getType();

    if (!T->isObjCObjectPointerType() ||
        Ivar->hasAttr<IBOutletAttr>() || // Skip IBOutlets.
        Ivar->hasAttr<IBOutletCollectionAttr>()) // Skip IBOutletCollections.
      continue;

    containsPointerIvar = true;
    break;
  }

  if (!containsPointerIvar)
    return;

  // Determine if the class subclasses NSObject.
  IdentifierInfo* NSObjectII = &Ctx.Idents.get("NSObject");
  IdentifierInfo* SenTestCaseII = &Ctx.Idents.get("SenTestCase");


  for ( ; ID ; ID = ID->getSuperClass()) {
    IdentifierInfo *II = ID->getIdentifier();

    if (II == NSObjectII)
      break;

    // FIXME: For now, ignore classes that subclass SenTestCase, as these don't
    // need to implement -dealloc.  They implement tear down in another way,
    // which we should try and catch later.
    //  http://llvm.org/bugs/show_bug.cgi?id=3187
    if (II == SenTestCaseII)
      return;
  }

  if (!ID)
    return;

  // Get the "dealloc" selector.
  IdentifierInfo* II = &Ctx.Idents.get("dealloc");
  Selector S = Ctx.Selectors.getSelector(0, &II);
  const ObjCMethodDecl *MD = nullptr;

  // Scan the instance methods for "dealloc".
  for (const auto *I : D->instance_methods()) {
    if (I->getSelector() == S) {
      MD = I;
      break;
    }
  }

  PathDiagnosticLocation DLoc =
    PathDiagnosticLocation::createBegin(D, BR.getSourceManager());

  if (!MD) { // No dealloc found.

    const char* name = LOpts.getGC() == LangOptions::NonGC
                       ? "missing -dealloc"
                       : "missing -dealloc (Hybrid MM, non-GC)";

    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << "Objective-C class '" << *D << "' lacks a 'dealloc' instance method";

    BR.EmitBasicReport(D, Checker, name, categories::CoreFoundationObjectiveC,
                       os.str(), DLoc);
    return;
  }

  // Get the "release" selector.
  IdentifierInfo* RII = &Ctx.Idents.get("release");
  Selector RS = Ctx.Selectors.getSelector(0, &RII);

  // Get the "self" identifier
  IdentifierInfo* SelfII = &Ctx.Idents.get("self");

  // Scan for missing and extra releases of ivars used by implementations
  // of synthesized properties
  for (const auto *I : D->property_impls()) {
    // We can only check the synthesized properties
    if (I->getPropertyImplementation() != ObjCPropertyImplDecl::Synthesize)
      continue;

    ObjCIvarDecl *ID = I->getPropertyIvarDecl();
    if (!ID)
      continue;

    QualType T = ID->getType();
    if (!T->isObjCObjectPointerType()) // Skip non-pointer ivars
      continue;

    const ObjCPropertyDecl *PD = I->getPropertyDecl();
    if (!PD)
      continue;

    // ivars cannot be set via read-only properties, so we'll skip them
    if (PD->isReadOnly())
      continue;

    // ivar must be released if and only if the kind of setter was not 'assign'
    bool requiresRelease = PD->getSetterKind() != ObjCPropertyDecl::Assign;
    if (scan_ivar_release(MD->getBody(), ID, PD, RS, SelfII, Ctx)
       != requiresRelease) {
      const char *name = nullptr;
      std::string buf;
      llvm::raw_string_ostream os(buf);

      if (requiresRelease) {
        name = LOpts.getGC() == LangOptions::NonGC
               ? "missing ivar release (leak)"
               : "missing ivar release (Hybrid MM, non-GC)";

        os << "The '" << *ID
           << "' instance variable was retained by a synthesized property but "
              "wasn't released in 'dealloc'";
      } else {
        name = LOpts.getGC() == LangOptions::NonGC
               ? "extra ivar release (use-after-release)"
               : "extra ivar release (Hybrid MM, non-GC)";

        os << "The '" << *ID
           << "' instance variable was not retained by a synthesized property "
              "but was released in 'dealloc'";
      }

      PathDiagnosticLocation SDLoc =
        PathDiagnosticLocation::createBegin(I, BR.getSourceManager());

      BR.EmitBasicReport(MD, Checker, name,
                         categories::CoreFoundationObjectiveC, os.str(), SDLoc);
    }
  }
}

//===----------------------------------------------------------------------===//
// ObjCDeallocChecker
//===----------------------------------------------------------------------===//

namespace {
class ObjCDeallocChecker : public Checker<
                                      check::ASTDecl<ObjCImplementationDecl> > {
public:
  void checkASTDecl(const ObjCImplementationDecl *D, AnalysisManager& mgr,
                    BugReporter &BR) const {
    if (mgr.getLangOpts().getGC() == LangOptions::GCOnly)
      return;
    checkObjCDealloc(this, cast<ObjCImplementationDecl>(D), mgr.getLangOpts(),
                     BR);
  }
};
}

void ento::registerObjCDeallocChecker(CheckerManager &mgr) {
  mgr.registerChecker<ObjCDeallocChecker>();
}
