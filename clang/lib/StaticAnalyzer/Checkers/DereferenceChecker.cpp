//===-- DereferenceChecker.cpp - Null dereference checker -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines NullDerefChecker, a builtin check in ExprEngine that performs
// checks for null pointers at loads and stores.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExprObjC.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

namespace {

class DerefBugType : public BugType {
  StringRef ArrayMsg, FieldMsg;

public:
  DerefBugType(CheckerFrontend *FE, StringRef Desc, const char *AMsg,
               const char *FMsg = nullptr)
      : BugType(FE, Desc), ArrayMsg(AMsg), FieldMsg(FMsg ? FMsg : AMsg) {}
  StringRef getArrayMsg() const { return ArrayMsg; }
  StringRef getFieldMsg() const { return FieldMsg; }
};

class DereferenceChecker
    : public CheckerFamily<check::Location, check::Bind,
                           check::PreStmt<BinaryOperator>,
                           EventDispatcher<ImplicitNullDerefEvent>> {
  void reportDerefBug(const DerefBugType &BT, ProgramStateRef State,
                      const Stmt *S, CheckerContext &C) const;

  bool suppressReport(CheckerContext &C, const Expr *E) const;

public:
  void checkLocation(SVal location, bool isLoad, const Stmt* S,
                     CheckerContext &C) const;
  void checkBind(SVal L, SVal V, const Stmt *S, bool AtDeclInit,
                 CheckerContext &C) const;
  void checkPreStmt(const BinaryOperator *Op, CheckerContext &C) const;

  static void AddDerefSource(raw_ostream &os,
                             SmallVectorImpl<SourceRange> &Ranges,
                             const Expr *Ex, const ProgramState *state,
                             const LocationContext *LCtx,
                             bool loadedFrom = false);

  CheckerFrontend NullDerefChecker, FixedDerefChecker, NullPointerArithmChecker;
  const DerefBugType NullBug{&NullDerefChecker, "Dereference of null pointer",
                             "a null pointer dereference",
                             "a dereference of a null pointer"};
  const DerefBugType UndefBug{&NullDerefChecker,
                              "Dereference of undefined pointer value",
                              "an undefined pointer dereference",
                              "a dereference of an undefined pointer value"};
  const DerefBugType LabelBug{&NullDerefChecker,
                              "Dereference of the address of a label",
                              "an undefined pointer dereference",
                              "a dereference of an address of a label"};
  const DerefBugType FixedAddressBug{&FixedDerefChecker,
                                     "Dereference of a fixed address",
                                     "a dereference of a fixed address"};
  const BugType NullPointerArithmBug{
      &NullPointerArithmChecker,
      "Possible undefined arithmetic operation involving a null pointer"};

  StringRef getDebugTag() const override { return "DereferenceChecker"; }
};
} // end anonymous namespace

void
DereferenceChecker::AddDerefSource(raw_ostream &os,
                                   SmallVectorImpl<SourceRange> &Ranges,
                                   const Expr *Ex,
                                   const ProgramState *state,
                                   const LocationContext *LCtx,
                                   bool loadedFrom) {
  Ex = Ex->IgnoreParenLValueCasts();
  switch (Ex->getStmtClass()) {
    default:
      break;
    case Stmt::DeclRefExprClass: {
      const DeclRefExpr *DR = cast<DeclRefExpr>(Ex);
      if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        os << " (" << (loadedFrom ? "loaded from" : "from")
           << " variable '" <<  VD->getName() << "')";
        Ranges.push_back(DR->getSourceRange());
      }
      break;
    }
    case Stmt::MemberExprClass: {
      const MemberExpr *ME = cast<MemberExpr>(Ex);
      os << " (" << (loadedFrom ? "loaded from" : "via")
         << " field '" << ME->getMemberNameInfo() << "')";
      SourceLocation L = ME->getMemberLoc();
      Ranges.push_back(SourceRange(L, L));
      break;
    }
    case Stmt::ObjCIvarRefExprClass: {
      const ObjCIvarRefExpr *IV = cast<ObjCIvarRefExpr>(Ex);
      os << " (" << (loadedFrom ? "loaded from" : "via")
         << " ivar '" << IV->getDecl()->getName() << "')";
      SourceLocation L = IV->getLocation();
      Ranges.push_back(SourceRange(L, L));
      break;
    }
  }
}

static const Expr *getDereferenceExpr(const Stmt *S, bool IsBind=false){
  const Expr *E = nullptr;

  // Walk through lvalue casts to get the original expression
  // that syntactically caused the load.
  if (const Expr *expr = dyn_cast<Expr>(S))
    E = expr->IgnoreParenLValueCasts();

  if (IsBind) {
    const VarDecl *VD;
    const Expr *Init;
    std::tie(VD, Init) = parseAssignment(S);
    if (VD && Init)
      E = Init;
  }
  return E;
}

bool DereferenceChecker::suppressReport(CheckerContext &C,
                                        const Expr *E) const {
  // Do not report dereferences on memory that use address space #256, #257,
  // and #258. Those address spaces are used when dereferencing address spaces
  // relative to the GS, FS, and SS segments on x86/x86-64 targets.
  // Dereferencing a null pointer in these address spaces is not defined
  // as an error. All other null dereferences in other address spaces
  // are defined as an error unless explicitly defined.
  // See https://clang.llvm.org/docs/LanguageExtensions.html, the section
  // "X86/X86-64 Language Extensions"

  QualType Ty = E->getType();
  if (!Ty.hasAddressSpace())
    return false;
  if (C.getAnalysisManager()
          .getAnalyzerOptions()
          .ShouldSuppressAddressSpaceDereferences)
    return true;

  const llvm::Triple::ArchType Arch =
      C.getASTContext().getTargetInfo().getTriple().getArch();

  if ((Arch == llvm::Triple::x86) || (Arch == llvm::Triple::x86_64)) {
    switch (toTargetAddressSpace(E->getType().getAddressSpace())) {
    case 256:
    case 257:
    case 258:
      return true;
    }
  }
  return false;
}

static bool isDeclRefExprToReference(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl()->getType()->isReferenceType();
  return false;
}

void DereferenceChecker::reportDerefBug(const DerefBugType &BT,
                                        ProgramStateRef State, const Stmt *S,
                                        CheckerContext &C) const {
  assert(&BT != &NullPointerArithmBug && "Invalid use of function");

  if (&BT == &FixedAddressBug) {
    if (!FixedDerefChecker.isEnabled())
      // Deliberately don't add a sink node if check is disabled.
      // This situation may be valid in special cases.
      return;
  } else {
    if (!NullDerefChecker.isEnabled()) {
      C.addSink();
      return;
    }
  }

  // Generate an error node.
  ExplodedNode *N = C.generateErrorNode(State);
  if (!N)
    return;

  SmallString<100> Buf;
  llvm::raw_svector_ostream Out(Buf);

  SmallVector<SourceRange, 2> Ranges;

  switch (S->getStmtClass()) {
  case Stmt::ArraySubscriptExprClass: {
    Out << "Array access";
    const ArraySubscriptExpr *AE = cast<ArraySubscriptExpr>(S);
    AddDerefSource(Out, Ranges, AE->getBase()->IgnoreParenCasts(), State.get(),
                   N->getLocationContext());
    Out << " results in " << BT.getArrayMsg();
    break;
  }
  case Stmt::ArraySectionExprClass: {
    Out << "Array access";
    const ArraySectionExpr *AE = cast<ArraySectionExpr>(S);
    AddDerefSource(Out, Ranges, AE->getBase()->IgnoreParenCasts(), State.get(),
                   N->getLocationContext());
    Out << " results in " << BT.getArrayMsg();
    break;
  }
  case Stmt::UnaryOperatorClass: {
    Out << BT.getDescription();
    const UnaryOperator *U = cast<UnaryOperator>(S);
    AddDerefSource(Out, Ranges, U->getSubExpr()->IgnoreParens(), State.get(),
                   N->getLocationContext(), true);
    break;
  }
  case Stmt::MemberExprClass: {
    const MemberExpr *M = cast<MemberExpr>(S);
    if (M->isArrow() || isDeclRefExprToReference(M->getBase())) {
      Out << "Access to field '" << M->getMemberNameInfo() << "' results in "
          << BT.getFieldMsg();
      AddDerefSource(Out, Ranges, M->getBase()->IgnoreParenCasts(), State.get(),
                     N->getLocationContext(), true);
    }
    break;
  }
  case Stmt::ObjCIvarRefExprClass: {
    const ObjCIvarRefExpr *IV = cast<ObjCIvarRefExpr>(S);
    Out << "Access to instance variable '" << *IV->getDecl() << "' results in "
        << BT.getFieldMsg();
    AddDerefSource(Out, Ranges, IV->getBase()->IgnoreParenCasts(), State.get(),
                   N->getLocationContext(), true);
    break;
  }
  default:
    break;
  }

  auto BR = std::make_unique<PathSensitiveBugReport>(
      BT, Buf.empty() ? BT.getDescription() : Buf.str(), N);

  bugreporter::trackExpressionValue(N, bugreporter::getDerefExpr(S), *BR);

  for (SmallVectorImpl<SourceRange>::iterator
       I = Ranges.begin(), E = Ranges.end(); I!=E; ++I)
    BR->addRange(*I);

  C.emitReport(std::move(BR));
}

void DereferenceChecker::checkLocation(SVal l, bool isLoad, const Stmt* S,
                                       CheckerContext &C) const {
  // Check for dereference of an undefined value.
  if (l.isUndef()) {
    const Expr *DerefExpr = getDereferenceExpr(S);
    if (!suppressReport(C, DerefExpr))
      reportDerefBug(UndefBug, C.getState(), DerefExpr, C);
    return;
  }

  DefinedOrUnknownSVal location = l.castAs<DefinedOrUnknownSVal>();

  // Check for null dereferences.
  if (!isa<Loc>(location))
    return;

  ProgramStateRef state = C.getState();

  ProgramStateRef notNullState, nullState;
  std::tie(notNullState, nullState) = state->assume(location);

  if (nullState) {
    if (!notNullState) {
      // We know that 'location' can only be null.  This is what
      // we call an "explicit" null dereference.
      const Expr *expr = getDereferenceExpr(S);
      if (!suppressReport(C, expr)) {
        reportDerefBug(NullBug, nullState, expr, C);
        return;
      }
    }

    // Otherwise, we have the case where the location could either be
    // null or not-null.  Record the error node as an "implicit" null
    // dereference.
    if (ExplodedNode *N = C.generateSink(nullState, C.getPredecessor())) {
      ImplicitNullDerefEvent event = {l, isLoad, N, &C.getBugReporter(),
                                      /*IsDirectDereference=*/true};
      dispatchEvent(event);
    }
  }

  if (location.isConstant()) {
    const Expr *DerefExpr = getDereferenceExpr(S, isLoad);
    if (!suppressReport(C, DerefExpr))
      reportDerefBug(FixedAddressBug, notNullState, DerefExpr, C);
    return;
  }

  // From this point forward, we know that the location is not null.
  C.addTransition(notNullState);
}

void DereferenceChecker::checkBind(SVal L, SVal V, const Stmt *S,
                                   bool AtDeclInit, CheckerContext &C) const {
  // If we're binding to a reference, check if the value is known to be null.
  if (V.isUndef())
    return;

  // One should never write to label addresses.
  if (auto Label = L.getAs<loc::GotoLabel>()) {
    reportDerefBug(LabelBug, C.getState(), S, C);
    return;
  }

  const MemRegion *MR = L.getAsRegion();
  const TypedValueRegion *TVR = dyn_cast_or_null<TypedValueRegion>(MR);
  if (!TVR)
    return;

  if (!TVR->getValueType()->isReferenceType())
    return;

  ProgramStateRef State = C.getState();

  ProgramStateRef StNonNull, StNull;
  std::tie(StNonNull, StNull) = State->assume(V.castAs<DefinedOrUnknownSVal>());

  if (StNull) {
    if (!StNonNull) {
      const Expr *expr = getDereferenceExpr(S, /*IsBind=*/true);
      if (!suppressReport(C, expr)) {
        reportDerefBug(NullBug, StNull, expr, C);
        return;
      }
    }

    // At this point the value could be either null or non-null.
    // Record this as an "implicit" null dereference.
    if (ExplodedNode *N = C.generateSink(StNull, C.getPredecessor())) {
      ImplicitNullDerefEvent event = {V, /*isLoad=*/true, N,
                                      &C.getBugReporter(),
                                      /*IsDirectDereference=*/true};
      dispatchEvent(event);
    }
  }

  if (V.isConstant()) {
    const Expr *DerefExpr = getDereferenceExpr(S, true);
    if (!suppressReport(C, DerefExpr))
      reportDerefBug(FixedAddressBug, State, DerefExpr, C);
    return;
  }

  // Unlike a regular null dereference, initializing a reference with a
  // dereferenced null pointer does not actually cause a runtime exception in
  // Clang's implementation of references.
  //
  //   int &r = *p; // safe??
  //   if (p != NULL) return; // uh-oh
  //   r = 5; // trap here
  //
  // The standard says this is invalid as soon as we try to create a "null
  // reference" (there is no such thing), but turning this into an assumption
  // that 'p' is never null will not match our actual runtime behavior.
  // So we do not record this assumption, allowing us to warn on the last line
  // of this example.
  //
  // We do need to add a transition because we may have generated a sink for
  // the "implicit" null dereference.
  C.addTransition(State, this);
}

void DereferenceChecker::checkPreStmt(const BinaryOperator *Op,
                                      CheckerContext &C) const {
  if (!Op->isAdditiveOp())
    return;
  const Expr *E1 = Op->getLHS();
  const Expr *E2 = Op->getRHS();
  QualType T1 = E1->getType().getCanonicalType();
  QualType T2 = E2->getType().getCanonicalType();
  if (T1->isIntegerType() && T2->isIntegerType())
    return;
  if (!T1->isPointerType() && !T1->isIntegerType() && !T2->isPointerType() &&
      !T2->isIntegerType())
    return;

  ProgramStateRef State = C.getState();
  SVal V1 = State->getSVal(E1, C.getLocationContext());
  SVal V2 = State->getSVal(E2, C.getLocationContext());
  if (V1.isUndef() || V2.isUndef())
    return;

  ConditionTruthVal V1IsNull = State->isNull(V1);
  ConditionTruthVal V2IsNull = State->isNull(V2);
  bool IsConstrained = true;

  // Check cases 'NULL + x' and 'NULL - x'
  if (T1->isPointerType() && T2->isIntegerType()) {
    if (!V1IsNull.isConstrainedTrue() || V2IsNull.isConstrainedTrue())
      return;
    IsConstrained = V2IsNull.isConstrainedFalse();
  }

  // Check case 'x + NULL'
  if (T1->isIntegerType() && T2->isPointerType()) {
    if (V1IsNull.isConstrainedTrue() || !V2IsNull.isConstrainedTrue())
      return;
    IsConstrained = V1IsNull.isConstrainedFalse();
  }

  // Check case 'NULL - p' or 'p - NULL'
  if (T1->isPointerType() && T2->isPointerType()) {
    if (!V1IsNull.isConstrainedTrue() && !V2IsNull.isConstrainedTrue())
      return;
    if (V1IsNull.isConstrainedTrue() && V2IsNull.isConstrainedTrue())
      return;
    IsConstrained =
        V1IsNull.isConstrainedFalse() || V2IsNull.isConstrainedFalse();
  }

  SmallString<100> Buf;
  llvm::raw_svector_ostream Out(Buf);
  SmallVector<SourceRange, 2> Ranges;

  auto AddSubExprStr = [&](const Expr *E, bool IsPointer,
                           ConditionTruthVal IsNull) {
    if (IsNull.isConstrainedTrue()) {
      if (IsPointer)
        Out << "null pointer";
      else
        Out << "zero";
    } else {
      if (!IsNull.isConstrainedFalse())
        Out << "probably ";
      if (IsPointer)
        Out << "non-null pointer";
      else
        Out << "nonzero integer value";
    }
    if (IsPointer)
      AddDerefSource(Out, Ranges, E, State.get(), C.getLocationContext(),
                     false);
  };

  if (Op->getOpcode() == BO_Add)
    Out << "Addition of a ";
  else
    Out << "Subtraction of a ";
  AddSubExprStr(E1, T1->isPointerType(), V1IsNull);
  Out << " and a ";
  AddSubExprStr(E2, T2->isPointerType(), V2IsNull);

  if (IsConstrained)
    Out << " results ";
  else
    Out << " may result ";
  Out << "in undefined behavior";

  ExplodedNode *N = C.generateErrorNode(State);
  if (!N)
    return;
  auto BR = std::make_unique<PathSensitiveBugReport>(NullPointerArithmBug,
                                                     Buf.str(), N);

  if (T1->isPointerType())
    bugreporter::trackExpressionValue(N, E1, *BR);
  if (T2->isPointerType())
    bugreporter::trackExpressionValue(N, E2, *BR);

  for (SmallVectorImpl<SourceRange>::iterator I = Ranges.begin(),
                                              E = Ranges.end();
       I != E; ++I)
    BR->addRange(*I);

  C.emitReport(std::move(BR));
}

void ento::registerNullDereferenceChecker(CheckerManager &Mgr) {
  Mgr.getChecker<DereferenceChecker>()->NullDerefChecker.enable(Mgr);
}

bool ento::shouldRegisterNullDereferenceChecker(const CheckerManager &) {
  return true;
}

void ento::registerFixedAddressDereferenceChecker(CheckerManager &Mgr) {
  Mgr.getChecker<DereferenceChecker>()->FixedDerefChecker.enable(Mgr);
}

bool ento::shouldRegisterFixedAddressDereferenceChecker(
    const CheckerManager &) {
  return true;
}

void ento::registerNullPointerArithmChecker(CheckerManager &Mgr) {
  Mgr.getChecker<DereferenceChecker>()->NullPointerArithmChecker.enable(Mgr);
}

bool ento::shouldRegisterNullPointerArithmChecker(const CheckerManager &) {
  return true;
}
