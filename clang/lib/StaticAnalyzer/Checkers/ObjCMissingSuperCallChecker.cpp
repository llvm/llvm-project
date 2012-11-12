//==- ObjCMissingSuperCallChecker.cpp - Check missing super-calls in ObjC --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a ObjCMissingSuperCallChecker, a checker that
//  analyzes a UIViewController implementation to determine if it
//  correctly calls super in the methods where this is mandatory.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

static bool isUIViewControllerSubclass(ASTContext &Ctx, 
                                       const ObjCImplementationDecl *D) {
  IdentifierInfo *ViewControllerII = &Ctx.Idents.get("UIViewController");
  const ObjCInterfaceDecl *ID = D->getClassInterface();

  for ( ; ID; ID = ID->getSuperClass())
    if (ID->getIdentifier() == ViewControllerII)
      return true;
  return false;  
}

//===----------------------------------------------------------------------===//
// FindSuperCallVisitor - Identify specific calls to the superclass.
//===----------------------------------------------------------------------===//

class FindSuperCallVisitor : public RecursiveASTVisitor<FindSuperCallVisitor> {
public:
  explicit FindSuperCallVisitor(Selector S) : DoesCallSuper(false), Sel(S) {}

  bool VisitObjCMessageExpr(ObjCMessageExpr *E) {
    if (E->getSelector() == Sel)
      if (E->getReceiverKind() == ObjCMessageExpr::SuperInstance)
        DoesCallSuper = true;

    // Recurse if we didn't find the super call yet.
    return !DoesCallSuper; 
  }

  bool DoesCallSuper;

private:
  Selector Sel;
};

//===----------------------------------------------------------------------===//
// ObjCSuperCallChecker 
//===----------------------------------------------------------------------===//

namespace {
class ObjCSuperCallChecker : public Checker<
                                      check::ASTDecl<ObjCImplementationDecl> > {
public:
  void checkASTDecl(const ObjCImplementationDecl *D, AnalysisManager &Mgr,
                    BugReporter &BR) const;
};
}

void ObjCSuperCallChecker::checkASTDecl(const ObjCImplementationDecl *D,
                                        AnalysisManager &Mgr,
                                        BugReporter &BR) const {
  ASTContext &Ctx = BR.getContext();

  if (!isUIViewControllerSubclass(Ctx, D))
    return;

  const char *SelectorNames[] = 
    {"addChildViewController", "viewDidAppear", "viewDidDisappear", 
     "viewWillAppear", "viewWillDisappear", "removeFromParentViewController",
     "didReceiveMemoryWarning", "viewDidUnload", "viewWillUnload",
     "viewDidLoad"};
  const unsigned SelectorArgumentCounts[] =
   {1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
  const size_t SelectorCount = llvm::array_lengthof(SelectorNames);
  assert(llvm::array_lengthof(SelectorArgumentCounts) == SelectorCount);

  // Fill the Selectors SmallSet with all selectors we want to check.
  llvm::SmallSet<Selector, 16> Selectors;
  for (size_t i = 0; i < SelectorCount; i++) { 
    unsigned ArgumentCount = SelectorArgumentCounts[i];
    const char *SelectorCString = SelectorNames[i];

    // Get the selector.
    IdentifierInfo *II = &Ctx.Idents.get(SelectorCString);
    Selectors.insert(Ctx.Selectors.getSelector(ArgumentCount, &II));
  }

  // Iterate over all instance methods.
  for (ObjCImplementationDecl::instmeth_iterator I = D->instmeth_begin(),
                                                 E = D->instmeth_end();
       I != E; ++I) {
    Selector S = (*I)->getSelector();
    // Find out whether this is a selector that we want to check.
    if (!Selectors.count(S))
      continue;

    ObjCMethodDecl *MD = *I;

    // Check if the method calls its superclass implementation.
    if (MD->getBody())
    {
      FindSuperCallVisitor Visitor(S);
      Visitor.TraverseDecl(MD);

      // It doesn't call super, emit a diagnostic.
      if (!Visitor.DoesCallSuper) {
        PathDiagnosticLocation DLoc =
          PathDiagnosticLocation::createEnd(MD->getBody(),
                                            BR.getSourceManager(),
                                            Mgr.getAnalysisDeclContext(D));

        const char *Name = "Missing call to superclass";
        SmallString<256> Buf;
        llvm::raw_svector_ostream os(Buf);

        os << "The '" << S.getAsString() 
           << "' instance method in UIViewController subclass '" << *D
           << "' is missing a [super " << S.getAsString() << "] call";

        BR.EmitBasicReport(MD, Name, categories::CoreFoundationObjectiveC,
                           os.str(), DLoc);
      }
    }
  }
}


//===----------------------------------------------------------------------===//
// Check registration.
//===----------------------------------------------------------------------===//

void ento::registerObjCSuperCallChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ObjCSuperCallChecker>();
}


/*
 ToDo list for expanding this check in the future, the list is not exhaustive.
 There are also cases where calling super is suggested but not "mandatory".
 In addition to be able to check the classes and methods below, architectural
 improvements like being able to allow for the super-call to be done in a called
 method would be good too.

*** trivial cases:
UIResponder subclasses
- resignFirstResponder

NSResponder subclasses
- cursorUpdate

*** more difficult cases:

UIDocument subclasses
- finishedHandlingError:recovered: (is multi-arg)
- finishedHandlingError:recovered: (is multi-arg)

UIViewController subclasses
- loadView (should *never* call super)
- transitionFromViewController:toViewController:
         duration:options:animations:completion: (is multi-arg)

UICollectionViewController subclasses
- loadView (take care because UIViewController subclasses should NOT call super
            in loadView, but UICollectionViewController subclasses should)

NSObject subclasses
- doesNotRecognizeSelector (it only has to call super if it doesn't throw)

UIPopoverBackgroundView subclasses (some of those are class methods)
- arrowDirection (should *never* call super)
- arrowOffset (should *never* call super)
- arrowBase (should *never* call super)
- arrowHeight (should *never* call super)
- contentViewInsets (should *never* call super)

UITextSelectionRect subclasses (some of those are properties)
- rect (should *never* call super)
- range (should *never* call super)
- writingDirection (should *never* call super)
- isVertical (should *never* call super)
- containsStart (should *never* call super)
- containsEnd (should *never* call super)
*/
