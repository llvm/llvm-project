//===- AnalysisDriver.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Support/ErrorBuilder.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <vector>

using namespace clang;
using namespace ssaf;

AnalysisDriver::AnalysisDriver(std::unique_ptr<LUSummary> LU)
    : LU(std::move(LU)) {}

llvm::Expected<std::vector<std::unique_ptr<AnalysisBase>>>
AnalysisDriver::toposort(llvm::ArrayRef<AnalysisName> Roots) {
  struct Visitor {
    enum class State { Unvisited, Visiting, Visited };

    std::map<AnalysisName, State> Marks;
    std::vector<AnalysisName> Path;
    std::vector<std::unique_ptr<AnalysisBase>> Result;

    explicit Visitor(size_t N) {
      Path.reserve(N);
      Result.reserve(N);
    }

    std::string formatCycle(const AnalysisName &CycleEntry) const {
      auto CycleBegin = llvm::find(Path, CycleEntry);
      std::string Cycle;
      llvm::raw_string_ostream OS(Cycle);
      llvm::interleave(llvm::make_range(CycleBegin, Path.end()), OS, " -> ");
      OS << " -> " << CycleEntry;
      return Cycle;
    }

    llvm::Error visit(const AnalysisName &Name) {
      auto [It, _] = Marks.emplace(Name, State::Unvisited);

      switch (It->second) {
      case State::Visited:
        return llvm::Error::success();

      case State::Visiting:
        return ErrorBuilder::create(std::errc::invalid_argument,
                                    "cycle detected: {0}", formatCycle(Name))
            .build();

      case State::Unvisited: {
        It->second = State::Visiting;
        Path.push_back(Name);

        llvm::Expected<std::unique_ptr<AnalysisBase>> V =
            AnalysisRegistry::instantiate(Name);
        if (!V) {
          return V.takeError();
        }

        // Unwrap for convenience to avoid the noise of dereferencing an
        // Expected on every subsequent access.
        std::unique_ptr<AnalysisBase> Analysis = std::move(*V);

        for (const auto &Dep : Analysis->dependencyNames()) {
          if (auto Err = visit(Dep)) {
            return Err;
          }
        }

        // std::map iterators are not invalidated by insertions, so It remains
        // valid after recursive visit() calls that insert new entries.
        It->second = State::Visited;
        Path.pop_back();
        Result.push_back(std::move(Analysis));

        return llvm::Error::success();
      }
      }
      llvm_unreachable("unhandled State");
    }
  };

  Visitor V(Roots.size());
  for (const auto &Root : Roots) {
    if (auto Err = V.visit(Root)) {
      return std::move(Err);
    }
  }
  return std::move(V.Result);
}

llvm::Error AnalysisDriver::executeSummaryAnalysis(SummaryAnalysisBase &Summary,
                                                   WPASuite &Suite) const {
  SummaryName SN = Summary.summaryName();
  auto DataIt = LU->Data.find(SN);
  if (DataIt == LU->Data.end()) {
    return ErrorBuilder::create(std::errc::invalid_argument,
                                "no data for analysis '{0}' in LUSummary",
                                Summary.analysisName())
        .build();
  }

  if (auto Err = Summary.initialize()) {
    return Err;
  }

  for (auto &[Id, EntitySummary] : DataIt->second) {
    if (auto Err = Summary.add(Id, *EntitySummary)) {
      return Err;
    }
  }

  if (auto Err = Summary.finalize()) {
    return Err;
  }

  return llvm::Error::success();
}

llvm::Error AnalysisDriver::executeDerivedAnalysis(DerivedAnalysisBase &Derived,
                                                   WPASuite &Suite) const {
  std::map<AnalysisName, const AnalysisResult *> DepMap;

  for (const auto &DepName : Derived.dependencyNames()) {
    auto It = Suite.Data.find(DepName);
    if (It == Suite.Data.end()) {
      ErrorBuilder::fatal("missing dependency '{0}' for analysis '{1}': "
                          "dependency graph is not topologically sorted",
                          DepName, Derived.analysisName());
    }
    DepMap[DepName] = It->second.get();
  }

  if (auto Err = Derived.initialize(DepMap)) {
    return Err;
  }

  while (true) {
    auto StepOrErr = Derived.step();
    if (!StepOrErr) {
      return StepOrErr.takeError();
    }
    if (!*StepOrErr) {
      break;
    }
  }

  if (auto Err = Derived.finalize()) {
    return Err;
  }

  return llvm::Error::success();
}

llvm::Expected<WPASuite> AnalysisDriver::execute(
    EntityIdTable IdTable,
    llvm::ArrayRef<std::unique_ptr<AnalysisBase>> Sorted) const {
  WPASuite Suite;
  Suite.IdTable = std::move(IdTable);

  for (auto &Analysis : Sorted) {
    switch (Analysis->TheKind) {
    case AnalysisBase::Kind::Summary: {
      SummaryAnalysisBase &SA = static_cast<SummaryAnalysisBase &>(*Analysis);
      if (auto Err = executeSummaryAnalysis(SA, Suite)) {
        return std::move(Err);
      }
      break;
    }
    case AnalysisBase::Kind::Derived: {
      DerivedAnalysisBase &DA = static_cast<DerivedAnalysisBase &>(*Analysis);
      if (auto Err = executeDerivedAnalysis(DA, Suite)) {
        return std::move(Err);
      }
      break;
    }
    }
    AnalysisName Name = Analysis->analysisName();
    Suite.Data.emplace(std::move(Name), std::move(*Analysis).result());
  }

  return std::move(Suite);
}

llvm::Expected<WPASuite> AnalysisDriver::run() && {
  auto ExpectedSorted = toposort(AnalysisRegistry::names());
  if (!ExpectedSorted) {
    return ExpectedSorted.takeError();
  }
  return execute(std::move(LU->IdTable), *ExpectedSorted);
}

llvm::Expected<WPASuite>
AnalysisDriver::run(llvm::ArrayRef<AnalysisName> Names) const {
  auto ExpectedSorted = toposort(Names);
  if (!ExpectedSorted) {
    return ExpectedSorted.takeError();
  }

  return execute(LU->IdTable, *ExpectedSorted);
}
