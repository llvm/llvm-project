//===-- DebugIteratorModeling.cpp ---------------------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a checker for debugging iterator modeling.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

#include "Iterator.h"

using namespace clang;
using namespace ento;
using namespace iterator;

namespace {

class DebugIteratorModeling
  : public Checker<eval::Call> {

  std::unique_ptr<BugType> DebugMsgBugType;

  template <typename Getter>
  void analyzerContainerDataField(const CallExpr *CE, CheckerContext &C,
                                  Getter get) const;
  void analyzerContainerBegin(const CallExpr *CE, CheckerContext &C) const;
  void analyzerContainerEnd(const CallExpr *CE, CheckerContext &C) const;
  template <typename Getter>
  void analyzerIteratorDataField(const CallExpr *CE, CheckerContext &C,
                                 Getter get, SVal Default) const;
  void analyzerIteratorPosition(const CallExpr *CE, CheckerContext &C) const;
  void analyzerIteratorContainer(const CallExpr *CE, CheckerContext &C) const;
  void analyzerIteratorValidity(const CallExpr *CE, CheckerContext &C) const;
  ExplodedNode *reportDebugMsg(llvm::StringRef Msg, CheckerContext &C) const;

  typedef void (DebugIteratorModeling::*FnCheck)(const CallExpr *,
                                                 CheckerContext &) const;

  CallDescriptionMap<FnCheck> Callbacks = {
    {{0, "clang_analyzer_container_begin", 1},
     &DebugIteratorModeling::analyzerContainerBegin},
    {{0, "clang_analyzer_container_end", 1},
     &DebugIteratorModeling::analyzerContainerEnd},
    {{0, "clang_analyzer_iterator_position", 1},
     &DebugIteratorModeling::analyzerIteratorPosition},
    {{0, "clang_analyzer_iterator_container", 1},
     &DebugIteratorModeling::analyzerIteratorContainer},
    {{0, "clang_analyzer_iterator_validity", 1},
     &DebugIteratorModeling::analyzerIteratorValidity},
  };

public:
  DebugIteratorModeling();

  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
};

} //namespace

DebugIteratorModeling::DebugIteratorModeling() {
  DebugMsgBugType.reset(
      new BugType(this, "Checking analyzer assumptions", "debug",
                  /*SuppressOnSink=*/true));
}

bool DebugIteratorModeling::evalCall(const CallEvent &Call,
                                     CheckerContext &C) const {
  const auto *CE = dyn_cast_or_null<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;

  const FnCheck *Handler = Callbacks.lookup(Call);
  if (!Handler)
    return false;

  (this->**Handler)(CE, C);
  return true;
}

template <typename Getter>
void DebugIteratorModeling::analyzerContainerDataField(const CallExpr *CE,
                                                       CheckerContext &C,
                                                       Getter get) const {
  if (CE->getNumArgs() == 0) {
    reportDebugMsg("Missing container argument", C);
    return;
  }

  auto State = C.getState();
  const MemRegion *Cont = C.getSVal(CE->getArg(0)).getAsRegion();
  if (Cont) {
    const auto *Data = getContainerData(State, Cont);
    if (Data) {
      SymbolRef Field = get(Data);
      if (Field) {
        State = State->BindExpr(CE, C.getLocationContext(),
                                nonloc::SymbolVal(Field));
        C.addTransition(State);
        return;
      }
    }
  }

  auto &BVF = C.getSValBuilder().getBasicValueFactory();
  State = State->BindExpr(CE, C.getLocationContext(),
                   nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(0))));
}

void DebugIteratorModeling::analyzerContainerBegin(const CallExpr *CE,
                                                   CheckerContext &C) const {
  analyzerContainerDataField(CE, C, [](const ContainerData *D) {
      return D->getBegin();
    });
}

void DebugIteratorModeling::analyzerContainerEnd(const CallExpr *CE,
                                                 CheckerContext &C) const {
  analyzerContainerDataField(CE, C, [](const ContainerData *D) {
      return D->getEnd();
    });
}

template <typename Getter>
void DebugIteratorModeling::analyzerIteratorDataField(const CallExpr *CE,
                                                      CheckerContext &C,
                                                      Getter get,
                                                      SVal Default) const {
  if (CE->getNumArgs() == 0) {
    reportDebugMsg("Missing iterator argument", C);
    return;
  }

  auto State = C.getState();
  SVal V = C.getSVal(CE->getArg(0));
  const auto *Pos = getIteratorPosition(State, V);
  if (Pos) {
    State = State->BindExpr(CE, C.getLocationContext(), get(Pos));
  } else {
    State = State->BindExpr(CE, C.getLocationContext(), Default);
  }
  C.addTransition(State);
}

void DebugIteratorModeling::analyzerIteratorPosition(const CallExpr *CE,
                                                     CheckerContext &C) const {
  auto &BVF = C.getSValBuilder().getBasicValueFactory();
  analyzerIteratorDataField(CE, C, [](const IteratorPosition *P) {
      return nonloc::SymbolVal(P->getOffset());
    }, nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(0))));
}

void DebugIteratorModeling::analyzerIteratorContainer(const CallExpr *CE,
                                                      CheckerContext &C) const {
  auto &BVF = C.getSValBuilder().getBasicValueFactory();
  analyzerIteratorDataField(CE, C, [](const IteratorPosition *P) {
      return loc::MemRegionVal(P->getContainer());
    }, loc::ConcreteInt(BVF.getValue(llvm::APSInt::get(0))));
}

void DebugIteratorModeling::analyzerIteratorValidity(const CallExpr *CE,
                                                     CheckerContext &C) const {
  auto &BVF = C.getSValBuilder().getBasicValueFactory();
  analyzerIteratorDataField(CE, C, [&BVF](const IteratorPosition *P) {
      return
        nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get((P->isValid()))));
    }, nonloc::ConcreteInt(BVF.getValue(llvm::APSInt::get(0))));
}

ExplodedNode *DebugIteratorModeling::reportDebugMsg(llvm::StringRef Msg,
                                                    CheckerContext &C) const {
  ExplodedNode *N = C.generateNonFatalErrorNode();
  if (!N)
    return nullptr;

  auto &BR = C.getBugReporter();
  BR.emitReport(std::make_unique<PathSensitiveBugReport>(*DebugMsgBugType,
                                                         Msg, N));
  return N;
}

void ento::registerDebugIteratorModeling(CheckerManager &mgr) {
  mgr.registerChecker<DebugIteratorModeling>();
}

bool ento::shouldRegisterDebugIteratorModeling(const LangOptions &LO) {
  return true;
}
