//===-- BlockInCriticalSectionChecker.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for blocks in critical sections. This checker should find
// the calls to blocking functions (for example: sleep, getc, fgets, read,
// recv etc.) inside a critical section. When sleep(x) is called while a mutex
// is held, other threades cannot lock the same mutex. This might take some
// time, leading to bad performance or even deadlock.
//
//===----------------------------------------------------------------------===//

#include "MutexModeling/MutexModelingAPI.h"

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

#include <utility>

using namespace clang;
using namespace ento;
using namespace mutex_modeling;

namespace {
class BlockInCriticalSectionChecker : public Checker<check::PostCall> {
private:
  const CallDescriptionSet BlockingFunctions{{CDM::CLibrary, {"sleep"}},
                                             {CDM::CLibrary, {"getc"}},
                                             {CDM::CLibrary, {"fgets"}},
                                             {CDM::CLibrary, {"read"}},
                                             {CDM::CLibrary, {"recv"}}};

  const BugType BlockInCritSectionBugType{
      this, "Call to blocking function in critical section", "Blocking Error"};

  [[nodiscard]] bool isBlockingInCritSection(const CallEvent &Call,
                                             CheckerContext &C) const;

  void reportBlockInCritSection(const CallEvent &call, CheckerContext &C) const;

public:
  BlockInCriticalSectionChecker();
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};

} // end anonymous namespace

BlockInCriticalSectionChecker::BlockInCriticalSectionChecker() {
  RegisterBugTypeForMutexModeling(&BlockInCritSectionBugType);
}

bool BlockInCriticalSectionChecker::isBlockingInCritSection(
    const CallEvent &Call, CheckerContext &C) const {
  return BlockingFunctions.contains(Call) && AreAnyCritsectionsActive(C);
}

void BlockInCriticalSectionChecker::reportBlockInCritSection(
    const CallEvent &Call, CheckerContext &C) const {
  ExplodedNode *ErrNode = C.generateNonFatalErrorNode(C.getState());
  if (!ErrNode)
    return;

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "Call to blocking function '" << Call.getCalleeIdentifier()->getName()
     << "' inside of critical section";
  auto R = std::make_unique<PathSensitiveBugReport>(BlockInCritSectionBugType,
                                                    os.str(), ErrNode);
  R->addRange(Call.getSourceRange());
  R->markInteresting(Call.getReturnValue());
  C.emitReport(std::move(R));
}

void BlockInCriticalSectionChecker::checkPostCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  if (!isBlockingInCritSection(Call, C))
    return;
  reportBlockInCritSection(Call, C);
}

// Checker registration
void ento::registerBlockInCriticalSectionChecker(CheckerManager &mgr) {
  mgr.registerChecker<BlockInCriticalSectionChecker>();

  // Register events for std::mutex lock and unlock
  // NOTE: There are standard library implementations where some methods
  // of `std::mutex` are inherited from an implementation detail base
  // class, and those aren't matched by the name specification {"std",
  // "mutex", "lock"}.
  // As a workaround here we omit the class name and only require the
  // presence of the name parts "std" and "lock"/"unlock".
  // TODO: Ensure that CallDescription understands inherited methods.
  RegisterEvent(EventDescriptor{
      mutex_modeling::MakeMemberExtractor({"std", /*"mutex"*/ "lock"}),
      EventKind::Acquire, LibraryKind::NotApplicable,
      SemanticsKind::XNUSemantics});
  RegisterEvent(EventDescriptor{
      mutex_modeling::MakeMemberExtractor({"std", /*"mutex"*/ "unlock"}),
      EventKind::Release});

  // Register events for std::lock_guard
  RegisterEvent(EventDescriptor{
      mutex_modeling::MakeRAIILockExtractor("lock_guard"), EventKind::Acquire,
      LibraryKind::NotApplicable, SemanticsKind::XNUSemantics});
  RegisterEvent(
      EventDescriptor{mutex_modeling::MakeRAIIReleaseExtractor("lock_guard"),
                      EventKind::Release});

  // Register events for std::unique_lock
  RegisterEvent(EventDescriptor{
      mutex_modeling::MakeRAIILockExtractor("unique_lock"), EventKind::Acquire,
      LibraryKind::NotApplicable, SemanticsKind::XNUSemantics});
  RegisterEvent(
      EventDescriptor{mutex_modeling::MakeRAIIReleaseExtractor("unique_lock"),
                      EventKind::Release});

  // Register events for std::scoped_lock
  RegisterEvent(EventDescriptor{
      mutex_modeling::MakeRAIILockExtractor("scoped_lock"), EventKind::Acquire,
      LibraryKind::NotApplicable, SemanticsKind::XNUSemantics});
  RegisterEvent(
      EventDescriptor{mutex_modeling::MakeRAIIReleaseExtractor("scoped_lock"),
                      EventKind::Release});
}

bool ento::shouldRegisterBlockInCriticalSectionChecker(
    const CheckerManager &mgr) {
  return true;
}
