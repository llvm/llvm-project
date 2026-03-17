//===- AnalysisDriver.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/SummaryAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <vector>

using namespace clang;
using namespace ssaf;

AnalysisDriver::AnalysisDriver(std::unique_ptr<LUSummary> LU)
    : LU(std::move(LU)) {}

llvm::Expected<std::vector<std::unique_ptr<AnalysisBase>>>
AnalysisDriver::sort(llvm::ArrayRef<AnalysisName> Roots) {
  struct Visitor {
    enum class State { Unvisited, Visiting, Visited };

    std::map<AnalysisName, State> Marks;
    std::vector<std::unique_ptr<AnalysisBase>> Result;

    llvm::Error visit(const AnalysisName &Name) {
      auto It = Marks.find(Name);
      switch (It != Marks.end() ? It->second : State::Unvisited) {
      case State::Visited:
        return llvm::Error::success();

      case State::Visiting:
        return ErrorBuilder::create(std::errc::invalid_argument,
                                    "cycle detected involving analysis '{0}'",
                                    Name)
            .build();

      case State::Unvisited: {
        Marks[Name] = State::Visiting;

        auto V = AnalysisRegistry::instantiate(Name.str());
        if (!V) {
          return V.takeError();
        }

        auto Analysis = std::move(*V);
        for (const auto &Dep : Analysis->dependencyNames()) {
          if (auto Err = visit(Dep)) {
            return Err;
          }
        }

        Marks[Name] = State::Visited;
        Result.push_back(std::move(Analysis));
        return llvm::Error::success();
      }
      }
      llvm_unreachable("unhandled State");
    }
  };

  Visitor V;
  for (const auto &Root : Roots) {
    if (auto Err = V.visit(Root)) {
      return std::move(Err);
    }
  }
  return std::move(V.Result);
}

llvm::Error AnalysisDriver::executeSummaryAnalysis(
    std::unique_ptr<SummaryAnalysisBase> Summary, WPASuite &Suite) {
  SummaryName SN = Summary->summaryName();
  auto DataIt = LU->Data.find(SN);
  if (DataIt == LU->Data.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                "no data for analysis '{0}' in LUSummary",
                                Summary->analysisName().str())
        .build();
  }

  if (auto Err = Summary->initialize()) {
    return Err;
  }

  for (auto &[Id, EntitySummary] : DataIt->second) {
    if (auto Err = Summary->add(Id, *EntitySummary)) {
      return Err;
    }
  }

  if (auto Err = Summary->finalize()) {
    return Err;
  }

  AnalysisName Name = Summary->analysisName();
  Suite.Data.emplace(Name, std::move(*Summary).result());

  return llvm::Error::success();
}

llvm::Error AnalysisDriver::executeDerivedAnalysis(
    std::unique_ptr<DerivedAnalysisBase> Derived, WPASuite &Suite) {
  std::map<AnalysisName, const AnalysisResult *> DepMap;

  for (const auto &DepName : Derived->dependencyNames()) {
    auto It = Suite.Data.find(DepName);
    if (It == Suite.Data.end()) {
      ErrorBuilder::fatal("missing dependency '{0}' for analysis '{1}': "
                          "dependency graph is not topologically sorted",
                          DepName.str(), Derived->analysisName().str());
    }
    DepMap[DepName] = It->second.get();
  }

  if (auto Err = Derived->initialize(DepMap)) {
    return Err;
  }

  while (true) {
    auto StepOrErr = Derived->step();
    if (!StepOrErr) {
      return StepOrErr.takeError();
    }
    if (!*StepOrErr) {
      break;
    }
  }

  if (auto Err = Derived->finalize()) {
    return Err;
  }

  AnalysisName Name = Derived->analysisName();
  Suite.Data.emplace(Name, std::move(*Derived).result());

  return llvm::Error::success();
}

llvm::Expected<WPASuite>
AnalysisDriver::execute(EntityIdTable IdTable,
                        std::vector<std::unique_ptr<AnalysisBase>> Sorted) {
  WPASuite Suite;
  Suite.IdTable = std::move(IdTable);

  for (auto &V : Sorted) {
    switch (V->TheKind) {
    case AnalysisBase::Kind::Summary: {
      auto SA = std::unique_ptr<SummaryAnalysisBase>(
          static_cast<SummaryAnalysisBase *>(V.release()));
      if (auto Err = executeSummaryAnalysis(std::move(SA), Suite)) {
        return std::move(Err);
      }
      break;
    }
    case AnalysisBase::Kind::Derived: {
      auto DA = std::unique_ptr<DerivedAnalysisBase>(
          static_cast<DerivedAnalysisBase *>(V.release()));
      if (auto Err = executeDerivedAnalysis(std::move(DA), Suite)) {
        return std::move(Err);
      }
      break;
    }
    }
  }

  return Suite;
}

llvm::Expected<WPASuite> AnalysisDriver::run() && {
  auto ExpectedSorted = sort(AnalysisRegistry::names());
  if (!ExpectedSorted) {
    return ExpectedSorted.takeError();
  }
  return execute(std::move(LU->IdTable), std::move(*ExpectedSorted));
}

llvm::Expected<WPASuite>
AnalysisDriver::run(llvm::ArrayRef<AnalysisName> Names) {
  auto ExpectedSorted = sort(Names);
  if (!ExpectedSorted) {
    return ExpectedSorted.takeError();
  }

  return execute(LU->IdTable, std::move(*ExpectedSorted));
}
