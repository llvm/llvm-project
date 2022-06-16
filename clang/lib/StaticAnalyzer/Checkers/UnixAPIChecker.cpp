//= UnixAPIChecker.h - Checks preconditions for various Unix APIs --*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines UnixAPIChecker, which is an assortment of checks on calls
// to various, widely used UNIX/Posix functions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace ento;

enum class OpenVariant {
  /// The standard open() call:
  ///    int open(const char *path, int oflag, ...);
  Open,

  /// The variant taking a directory file descriptor and a relative path:
  ///    int openat(int fd, const char *path, int oflag, ...);
  OpenAt
};

namespace {

class UnixAPIMisuseChecker : public Checker< check::PreStmt<CallExpr> > {
  mutable std::unique_ptr<BugType> BT_open, BT_pthreadOnce;
  mutable Optional<uint64_t> Val_O_CREAT;

public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;

  void CheckOpen(CheckerContext &C, const CallExpr *CE) const;
  void CheckOpenAt(CheckerContext &C, const CallExpr *CE) const;
  void CheckPthreadOnce(CheckerContext &C, const CallExpr *CE) const;

  void CheckOpenVariant(CheckerContext &C,
                        const CallExpr *CE, OpenVariant Variant) const;

  void ReportOpenBug(CheckerContext &C,
                     ProgramStateRef State,
                     const char *Msg,
                     SourceRange SR) const;

};

class UnixAPIPortabilityChecker : public Checker< check::PreStmt<CallExpr> > {
public:
  void checkPreStmt(const CallExpr *CE, CheckerContext &C) const;

private:
  mutable std::unique_ptr<BugType> BT_mallocZero;

  void CheckCallocZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckMallocZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckReallocZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckReallocfZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckAllocaZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckAllocaWithAlignZero(CheckerContext &C, const CallExpr *CE) const;
  void CheckVallocZero(CheckerContext &C, const CallExpr *CE) const;

  bool ReportZeroByteAllocation(CheckerContext &C,
                                ProgramStateRef falseState,
                                const Expr *arg,
                                const char *fn_name) const;
  void BasicAllocationCheck(CheckerContext &C,
                            const CallExpr *CE,
                            const unsigned numArgs,
                            const unsigned sizeArg,
                            const char *fn) const;
};

} //end anonymous namespace

static void LazyInitialize(const CheckerBase *Checker,
                           std::unique_ptr<BugType> &BT,
                           const char *name) {
  if (BT)
    return;
  BT.reset(new BugType(Checker, name, categories::UnixAPI));
}

//===----------------------------------------------------------------------===//
// "open" (man 2 open)
//===----------------------------------------------------------------------===/

void UnixAPIMisuseChecker::checkPreStmt(const CallExpr *CE,
                                        CheckerContext &C) const {
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD || FD->getKind() != Decl::Function)
    return;

  // Don't treat functions in namespaces with the same name a Unix function
  // as a call to the Unix function.
  const DeclContext *NamespaceCtx = FD->getEnclosingNamespaceContext();
  if (isa_and_nonnull<NamespaceDecl>(NamespaceCtx))
    return;

  StringRef FName = C.getCalleeName(FD);
  if (FName.empty())
    return;

  if (FName == "open")
    CheckOpen(C, CE);

  else if (FName == "openat")
    CheckOpenAt(C, CE);

  else if (FName == "pthread_once")
    CheckPthreadOnce(C, CE);
}
void UnixAPIMisuseChecker::ReportOpenBug(CheckerContext &C,
                                         ProgramStateRef State,
                                         const char *Msg,
                                         SourceRange SR) const {
  ExplodedNode *N = C.generateErrorNode(State);
  if (!N)
    return;

  LazyInitialize(this, BT_open, "Improper use of 'open'");

  auto Report = std::make_unique<PathSensitiveBugReport>(*BT_open, Msg, N);
  Report->addRange(SR);
  C.emitReport(std::move(Report));
}

void UnixAPIMisuseChecker::CheckOpen(CheckerContext &C,
                                     const CallExpr *CE) const {
  CheckOpenVariant(C, CE, OpenVariant::Open);
}

void UnixAPIMisuseChecker::CheckOpenAt(CheckerContext &C,
                                       const CallExpr *CE) const {
  CheckOpenVariant(C, CE, OpenVariant::OpenAt);
}

void UnixAPIMisuseChecker::CheckOpenVariant(CheckerContext &C,
                                            const CallExpr *CE,
                                            OpenVariant Variant) const {
  // The index of the argument taking the flags open flags (O_RDONLY,
  // O_WRONLY, O_CREAT, etc.),
  unsigned int FlagsArgIndex;
  const char *VariantName;
  switch (Variant) {
  case OpenVariant::Open:
    FlagsArgIndex = 1;
    VariantName = "open";
    break;
  case OpenVariant::OpenAt:
    FlagsArgIndex = 2;
    VariantName = "openat";
    break;
  };

  // All calls should at least provide arguments up to the 'flags' parameter.
  unsigned int MinArgCount = FlagsArgIndex + 1;

  // If the flags has O_CREAT set then open/openat() require an additional
  // argument specifying the file mode (permission bits) for the created file.
  unsigned int CreateModeArgIndex = FlagsArgIndex + 1;

  // The create mode argument should be the last argument.
  unsigned int MaxArgCount = CreateModeArgIndex + 1;

  ProgramStateRef state = C.getState();

  if (CE->getNumArgs() < MinArgCount) {
    // The frontend should issue a warning for this case. Just return.
    return;
  } else if (CE->getNumArgs() == MaxArgCount) {
    const Expr *Arg = CE->getArg(CreateModeArgIndex);
    QualType QT = Arg->getType();
    if (!QT->isIntegerType()) {
      SmallString<256> SBuf;
      llvm::raw_svector_ostream OS(SBuf);
      OS << "The " << CreateModeArgIndex + 1
         << llvm::getOrdinalSuffix(CreateModeArgIndex + 1)
         << " argument to '" << VariantName << "' is not an integer";

      ReportOpenBug(C, state,
                    SBuf.c_str(),
                    Arg->getSourceRange());
      return;
    }
  } else if (CE->getNumArgs() > MaxArgCount) {
    SmallString<256> SBuf;
    llvm::raw_svector_ostream OS(SBuf);
    OS << "Call to '" << VariantName << "' with more than " << MaxArgCount
       << " arguments";

    ReportOpenBug(C, state,
                  SBuf.c_str(),
                  CE->getArg(MaxArgCount)->getSourceRange());
    return;
  }

  // The definition of O_CREAT is platform specific.  We need a better way
  // of querying this information from the checking environment.
  if (!Val_O_CREAT.hasValue()) {
    if (C.getASTContext().getTargetInfo().getTriple().getVendor()
                                                      == llvm::Triple::Apple)
      Val_O_CREAT = 0x0200;
    else {
      // FIXME: We need a more general way of getting the O_CREAT value.
      // We could possibly grovel through the preprocessor state, but
      // that would require passing the Preprocessor object to the ExprEngine.
      // See also: MallocChecker.cpp / M_ZERO.
      return;
    }
  }

  // Now check if oflags has O_CREAT set.
  const Expr *oflagsEx = CE->getArg(FlagsArgIndex);
  const SVal V = C.getSVal(oflagsEx);
  if (!V.getAs<NonLoc>()) {
    // The case where 'V' can be a location can only be due to a bad header,
    // so in this case bail out.
    return;
  }
  NonLoc oflags = V.castAs<NonLoc>();
  NonLoc ocreateFlag = C.getSValBuilder()
      .makeIntVal(Val_O_CREAT.getValue(), oflagsEx->getType()).castAs<NonLoc>();
  SVal maskedFlagsUC = C.getSValBuilder().evalBinOpNN(state, BO_And,
                                                      oflags, ocreateFlag,
                                                      oflagsEx->getType());
  if (maskedFlagsUC.isUnknownOrUndef())
    return;
  DefinedSVal maskedFlags = maskedFlagsUC.castAs<DefinedSVal>();

  // Check if maskedFlags is non-zero.
  ProgramStateRef trueState, falseState;
  std::tie(trueState, falseState) = state->assume(maskedFlags);

  // Only emit an error if the value of 'maskedFlags' is properly
  // constrained;
  if (!(trueState && !falseState))
    return;

  if (CE->getNumArgs() < MaxArgCount) {
    SmallString<256> SBuf;
    llvm::raw_svector_ostream OS(SBuf);
    OS << "Call to '" << VariantName << "' requires a "
       << CreateModeArgIndex + 1
       << llvm::getOrdinalSuffix(CreateModeArgIndex + 1)
       << " argument when the 'O_CREAT' flag is set";
    ReportOpenBug(C, trueState,
                  SBuf.c_str(),
                  oflagsEx->getSourceRange());
  }
}

//===----------------------------------------------------------------------===//
// pthread_once
//===----------------------------------------------------------------------===//

void UnixAPIMisuseChecker::CheckPthreadOnce(CheckerContext &C,
                                      const CallExpr *CE) const {

  // This is similar to 'CheckDispatchOnce' in the MacOSXAPIChecker.
  // They can possibly be refactored.

  if (CE->getNumArgs() < 1)
    return;

  // Check if the first argument is stack allocated.  If so, issue a warning
  // because that's likely to be bad news.
  ProgramStateRef state = C.getState();
  const MemRegion *R = C.getSVal(CE->getArg(0)).getAsRegion();
  if (!R || !isa<StackSpaceRegion>(R->getMemorySpace()))
    return;

  ExplodedNode *N = C.generateErrorNode(state);
  if (!N)
    return;

  SmallString<256> S;
  llvm::raw_svector_ostream os(S);
  os << "Call to 'pthread_once' uses";
  if (const VarRegion *VR = dyn_cast<VarRegion>(R))
    os << " the local variable '" << VR->getDecl()->getName() << '\'';
  else
    os << " stack allocated memory";
  os << " for the \"control\" value.  Using such transient memory for "
  "the control value is potentially dangerous.";
  if (isa<VarRegion>(R) && isa<StackLocalsSpaceRegion>(R->getMemorySpace()))
    os << "  Perhaps you intended to declare the variable as 'static'?";

  LazyInitialize(this, BT_pthreadOnce, "Improper use of 'pthread_once'");

  auto report =
      std::make_unique<PathSensitiveBugReport>(*BT_pthreadOnce, os.str(), N);
  report->addRange(CE->getArg(0)->getSourceRange());
  C.emitReport(std::move(report));
}

//===----------------------------------------------------------------------===//
// "calloc", "malloc", "realloc", "reallocf", "alloca" and "valloc"
// with allocation size 0
//===----------------------------------------------------------------------===//

// FIXME: Eventually these should be rolled into the MallocChecker, but right now
// they're more basic and valuable for widespread use.

// Returns true if we try to do a zero byte allocation, false otherwise.
// Fills in trueState and falseState.
static bool IsZeroByteAllocation(ProgramStateRef state,
                                 const SVal argVal,
                                 ProgramStateRef *trueState,
                                 ProgramStateRef *falseState) {
  std::tie(*trueState, *falseState) =
    state->assume(argVal.castAs<DefinedSVal>());

  return (*falseState && !*trueState);
}

// Generates an error report, indicating that the function whose name is given
// will perform a zero byte allocation.
// Returns false if an error occurred, true otherwise.
bool UnixAPIPortabilityChecker::ReportZeroByteAllocation(
                                                    CheckerContext &C,
                                                    ProgramStateRef falseState,
                                                    const Expr *arg,
                                                    const char *fn_name) const {
  ExplodedNode *N = C.generateErrorNode(falseState);
  if (!N)
    return false;

  LazyInitialize(this, BT_mallocZero,
                 "Undefined allocation of 0 bytes (CERT MEM04-C; CWE-131)");

  SmallString<256> S;
  llvm::raw_svector_ostream os(S);
  os << "Call to '" << fn_name << "' has an allocation size of 0 bytes";
  auto report =
      std::make_unique<PathSensitiveBugReport>(*BT_mallocZero, os.str(), N);

  report->addRange(arg->getSourceRange());
  bugreporter::trackExpressionValue(N, arg, *report);
  C.emitReport(std::move(report));

  return true;
}

// Does a basic check for 0-sized allocations suitable for most of the below
// functions (modulo "calloc")
void UnixAPIPortabilityChecker::BasicAllocationCheck(CheckerContext &C,
                                                     const CallExpr *CE,
                                                     const unsigned numArgs,
                                                     const unsigned sizeArg,
                                                     const char *fn) const {
  // Check for the correct number of arguments.
  if (CE->getNumArgs() != numArgs)
    return;

  // Check if the allocation size is 0.
  ProgramStateRef state = C.getState();
  ProgramStateRef trueState = nullptr, falseState = nullptr;
  const Expr *arg = CE->getArg(sizeArg);
  SVal argVal = C.getSVal(arg);

  if (argVal.isUnknownOrUndef())
    return;

  // Is the value perfectly constrained to zero?
  if (IsZeroByteAllocation(state, argVal, &trueState, &falseState)) {
    (void) ReportZeroByteAllocation(C, falseState, arg, fn);
    return;
  }
  // Assume the value is non-zero going forward.
  assert(trueState);
  if (trueState != state)
    C.addTransition(trueState);
}

void UnixAPIPortabilityChecker::CheckCallocZero(CheckerContext &C,
                                                const CallExpr *CE) const {
  unsigned int nArgs = CE->getNumArgs();
  if (nArgs != 2)
    return;

  ProgramStateRef state = C.getState();
  ProgramStateRef trueState = nullptr, falseState = nullptr;

  unsigned int i;
  for (i = 0; i < nArgs; i++) {
    const Expr *arg = CE->getArg(i);
    SVal argVal = C.getSVal(arg);
    if (argVal.isUnknownOrUndef()) {
      if (i == 0)
        continue;
      else
        return;
    }

    if (IsZeroByteAllocation(state, argVal, &trueState, &falseState)) {
      if (ReportZeroByteAllocation(C, falseState, arg, "calloc"))
        return;
      else if (i == 0)
        continue;
      else
        return;
    }
  }

  // Assume the value is non-zero going forward.
  assert(trueState);
  if (trueState != state)
    C.addTransition(trueState);
}

void UnixAPIPortabilityChecker::CheckMallocZero(CheckerContext &C,
                                                const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 1, 0, "malloc");
}

void UnixAPIPortabilityChecker::CheckReallocZero(CheckerContext &C,
                                                 const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 2, 1, "realloc");
}

void UnixAPIPortabilityChecker::CheckReallocfZero(CheckerContext &C,
                                                  const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 2, 1, "reallocf");
}

void UnixAPIPortabilityChecker::CheckAllocaZero(CheckerContext &C,
                                                const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 1, 0, "alloca");
}

void UnixAPIPortabilityChecker::CheckAllocaWithAlignZero(
                                                     CheckerContext &C,
                                                     const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 2, 0, "__builtin_alloca_with_align");
}

void UnixAPIPortabilityChecker::CheckVallocZero(CheckerContext &C,
                                                const CallExpr *CE) const {
  BasicAllocationCheck(C, CE, 1, 0, "valloc");
}

void UnixAPIPortabilityChecker::checkPreStmt(const CallExpr *CE,
                                             CheckerContext &C) const {
  const FunctionDecl *FD = C.getCalleeDecl(CE);
  if (!FD || FD->getKind() != Decl::Function)
    return;

  // Don't treat functions in namespaces with the same name a Unix function
  // as a call to the Unix function.
  const DeclContext *NamespaceCtx = FD->getEnclosingNamespaceContext();
  if (isa_and_nonnull<NamespaceDecl>(NamespaceCtx))
    return;

  StringRef FName = C.getCalleeName(FD);
  if (FName.empty())
    return;

  if (FName == "calloc")
    CheckCallocZero(C, CE);

  else if (FName == "malloc")
    CheckMallocZero(C, CE);

  else if (FName == "realloc")
    CheckReallocZero(C, CE);

  else if (FName == "reallocf")
    CheckReallocfZero(C, CE);

  else if (FName == "alloca" || FName ==  "__builtin_alloca")
    CheckAllocaZero(C, CE);

  else if (FName == "__builtin_alloca_with_align")
    CheckAllocaWithAlignZero(C, CE);

  else if (FName == "valloc")
    CheckVallocZero(C, CE);
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

#define REGISTER_CHECKER(CHECKERNAME)                                          \
  void ento::register##CHECKERNAME(CheckerManager &mgr) {                      \
    mgr.registerChecker<CHECKERNAME>();                                        \
  }                                                                            \
                                                                               \
  bool ento::shouldRegister##CHECKERNAME(const CheckerManager &mgr) {          \
    return true;                                                               \
  }

REGISTER_CHECKER(UnixAPIMisuseChecker)
REGISTER_CHECKER(UnixAPIPortabilityChecker)
