//===- ReturnValueChecker - Applies guaranteed return values ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines ReturnValueChecker, which checks for calls with guaranteed
// boolean return value. It ensures the return value of each function call.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {
class ReturnValueChecker : public Checker<check::PostCall, check::EndFunction> {
public:
  // It sets the predefined invariant ('CDM') if the current call not break it.
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  // It reports whether a predefined invariant ('CDM') is broken.
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const;

private:
  // The pairs are in the following form: {{{class, call}}, return value}
  const CallDescriptionMap<bool> CDM = {
      // These are known in the LLVM project: 'Error()'
      {{{"ARMAsmParser", "Error"}}, true},
      {{{"HexagonAsmParser", "Error"}}, true},
      {{{"LLLexer", "Error"}}, true},
      {{{"LLParser", "Error"}}, true},
      {{{"MCAsmParser", "Error"}}, true},
      {{{"MCAsmParserExtension", "Error"}}, true},
      {{{"TGParser", "Error"}}, true},
      {{{"X86AsmParser", "Error"}}, true},
      // 'TokError()'
      {{{"LLParser", "TokError"}}, true},
      {{{"MCAsmParser", "TokError"}}, true},
      {{{"MCAsmParserExtension", "TokError"}}, true},
      {{{"TGParser", "TokError"}}, true},
      // 'error()'
      {{{"MIParser", "error"}}, true},
      {{{"WasmAsmParser", "error"}}, true},
      {{{"WebAssemblyAsmParser", "error"}}, true},
      // Other
      {{{"AsmParser", "printError"}}, true}};
};
} // namespace

static std::string getName(const CallEvent &Call) {
  std::string Name;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(Call.getDecl()))
    if (const CXXRecordDecl *RD = MD->getParent())
      Name += RD->getNameAsString() + "::";

  Name += Call.getCalleeIdentifier()->getName();
  return Name;
}

// The predefinitions ('CDM') could break due to the ever growing code base.
// Check for the expected invariants and see whether they apply.
static Optional<bool> isInvariantBreak(bool ExpectedValue, SVal ReturnV,
                                       CheckerContext &C) {
  auto ReturnDV = ReturnV.getAs<DefinedOrUnknownSVal>();
  if (!ReturnDV)
    return std::nullopt;

  if (ExpectedValue)
    return C.getState()->isNull(*ReturnDV).isConstrainedTrue();

  return C.getState()->isNull(*ReturnDV).isConstrainedFalse();
}

void ReturnValueChecker::checkPostCall(const CallEvent &Call,
                                       CheckerContext &C) const {
  const bool *RawExpectedValue = CDM.lookup(Call);
  if (!RawExpectedValue)
    return;

  SVal ReturnV = Call.getReturnValue();
  bool ExpectedValue = *RawExpectedValue;
  Optional<bool> IsInvariantBreak = isInvariantBreak(ExpectedValue, ReturnV, C);
  if (!IsInvariantBreak)
    return;

  // If the invariant is broken it is reported by 'checkEndFunction()'.
  if (*IsInvariantBreak)
    return;

  std::string Name = getName(Call);
  const NoteTag *CallTag = C.getNoteTag(
      [Name, ExpectedValue](PathSensitiveBugReport &) -> std::string {
        SmallString<128> Msg;
        llvm::raw_svector_ostream Out(Msg);

        Out << '\'' << Name << "' returns "
            << (ExpectedValue ? "true" : "false");
        return std::string(Out.str());
      },
      /*IsPrunable=*/true);

  ProgramStateRef State = C.getState();
  State = State->assume(ReturnV.castAs<DefinedOrUnknownSVal>(), ExpectedValue);
  C.addTransition(State, CallTag);
}

void ReturnValueChecker::checkEndFunction(const ReturnStmt *RS,
                                          CheckerContext &C) const {
  if (!RS || !RS->getRetValue())
    return;

  // We cannot get the caller in the top-frame.
  const StackFrameContext *SFC = C.getStackFrame();
  if (C.getStackFrame()->inTopFrame())
    return;

  ProgramStateRef State = C.getState();
  CallEventManager &CMgr = C.getStateManager().getCallEventManager();
  CallEventRef<> Call = CMgr.getCaller(SFC, State);
  if (!Call)
    return;

  const bool *RawExpectedValue = CDM.lookup(*Call);
  if (!RawExpectedValue)
    return;

  SVal ReturnV = State->getSVal(RS->getRetValue(), C.getLocationContext());
  bool ExpectedValue = *RawExpectedValue;
  Optional<bool> IsInvariantBreak = isInvariantBreak(ExpectedValue, ReturnV, C);
  if (!IsInvariantBreak)
    return;

  // If the invariant is appropriate it is reported by 'checkPostCall()'.
  if (!*IsInvariantBreak)
    return;

  std::string Name = getName(*Call);
  const NoteTag *CallTag = C.getNoteTag(
      [Name, ExpectedValue](BugReport &BR) -> std::string {
        SmallString<128> Msg;
        llvm::raw_svector_ostream Out(Msg);

        // The following is swapped because the invariant is broken.
        Out << '\'' << Name << "' returns "
            << (ExpectedValue ? "false" : "true");

        return std::string(Out.str());
      },
      /*IsPrunable=*/false);

  C.addTransition(State, CallTag);
}

void ento::registerReturnValueChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ReturnValueChecker>();
}

bool ento::shouldRegisterReturnValueChecker(const CheckerManager &mgr) {
  return true;
}
