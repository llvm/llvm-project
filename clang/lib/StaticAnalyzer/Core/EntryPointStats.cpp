//===- EntryPointStats.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/EntryPointStats.h"
#include "clang/AST/DeclBase.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <iterator>

using namespace clang;
using namespace ento;

namespace {
struct Registry {
  std::vector<UnsignedEPStat *> ExplicitlySetStats;
  std::vector<UnsignedMaxEPStat *> MaxStats;
  std::vector<CounterEPStat *> CounterStats;

  bool IsLocked = false;

  struct Snapshot {
    const Decl *EntryPoint;
    // Explicitly set statistics may not have a value set, so they are separate
    // from other unsigned statistics
    std::vector<std::optional<unsigned>> ExplicitlySetStatValues;
    // These are counting and maximizing statistics that initialize to 0, which
    // is meaningful even if they are never updated, so their value is always
    // present.
    std::vector<unsigned> MaxOrCountStatValues;

    void dumpAsCSV(llvm::raw_ostream &OS) const;
  };

  std::vector<Snapshot> Snapshots;
  std::string EscapedCPPFileName;
};
} // namespace

static llvm::ManagedStatic<Registry> StatsRegistry;

namespace {
template <typename Callback> void enumerateStatVectors(const Callback &Fn) {
  // This order is important, it matches the order of the Snapshot fields:
  // - ExplicitlySetStatValues
  Fn(StatsRegistry->ExplicitlySetStats);
  // - MaxOrCountStatValues
  Fn(StatsRegistry->MaxStats);
  Fn(StatsRegistry->CounterStats);
}

void clearSnapshots(void *) { StatsRegistry->Snapshots.clear(); }

} // namespace

static void checkStatName(const EntryPointStat *M) {
#ifdef NDEBUG
  return;
#endif // NDEBUG
  constexpr std::array AllowedSpecialChars = {
      '+', '-', '_', '=', ':', '(',  ')', '@', '!', '~',
      '$', '%', '^', '&', '*', '\'', ';', '<', '>', '/'};
  for (unsigned char C : M->name()) {
    if (!std::isalnum(C) && !llvm::is_contained(AllowedSpecialChars, C)) {
      llvm::errs() << "Stat name \"" << M->name() << "\" contains character '"
                   << C << "' (" << static_cast<int>(C)
                   << ") that is not allowed.";
      assert(false && "The Stat name contains unallowed character");
    }
  }
}

void EntryPointStat::lockRegistry(llvm::StringRef CPPFileName,
                                  ASTContext &Ctx) {
  auto CmpByNames = [](const EntryPointStat *L, const EntryPointStat *R) {
    return L->name() < R->name();
  };
  enumerateStatVectors(
      [CmpByNames](auto &Stats) { llvm::sort(Stats, CmpByNames); });
  enumerateStatVectors(
      [](const auto &Stats) { llvm::for_each(Stats, checkStatName); });
  StatsRegistry->IsLocked = true;
  llvm::raw_string_ostream OS(StatsRegistry->EscapedCPPFileName);
  llvm::printEscapedString(CPPFileName, OS);
  // Make sure snapshots (that reference function Decl's) do not persist after
  // the AST is destroyed. This is especially relevant in the context of unit
  // tests that construct and destruct multiple ASTs in the same process.
  Ctx.AddDeallocation(clearSnapshots, nullptr);
}

[[maybe_unused]] static bool isRegistered(llvm::StringLiteral Name) {
  auto ByName = [Name](const EntryPointStat *M) { return M->name() == Name; };
  bool Result = false;
  enumerateStatVectors([ByName, &Result](const auto &Stats) {
    Result = Result || llvm::any_of(Stats, ByName);
  });
  return Result;
}

CounterEPStat::CounterEPStat(llvm::StringLiteral Name) : EntryPointStat(Name) {
  assert(!StatsRegistry->IsLocked);
  assert(!isRegistered(Name));
  StatsRegistry->CounterStats.push_back(this);
}

UnsignedMaxEPStat::UnsignedMaxEPStat(llvm::StringLiteral Name)
    : EntryPointStat(Name) {
  assert(!StatsRegistry->IsLocked);
  assert(!isRegistered(Name));
  StatsRegistry->MaxStats.push_back(this);
}

UnsignedEPStat::UnsignedEPStat(llvm::StringLiteral Name)
    : EntryPointStat(Name) {
  assert(!StatsRegistry->IsLocked);
  assert(!isRegistered(Name));
  StatsRegistry->ExplicitlySetStats.push_back(this);
}

static std::vector<std::optional<unsigned>> consumeExplicitlySetStats() {
  std::vector<std::optional<unsigned>> Result;
  Result.reserve(StatsRegistry->ExplicitlySetStats.size());
  for (auto *M : StatsRegistry->ExplicitlySetStats) {
    Result.push_back(M->value());
    M->reset();
  }
  return Result;
}

static std::vector<unsigned> consumeMaxAndCounterStats() {
  std::vector<unsigned> Result;
  Result.reserve(StatsRegistry->CounterStats.size() +
                 StatsRegistry->MaxStats.size());
  // Order is important, it must match the order in enumerateStatVectors
  for (auto *M : StatsRegistry->MaxStats) {
    Result.push_back(M->value());
    M->reset();
  }
  for (auto *M : StatsRegistry->CounterStats) {
    Result.push_back(M->value());
    M->reset();
  }
  return Result;
}

static std::vector<llvm::StringLiteral> getStatNames() {
  std::vector<llvm::StringLiteral> Ret;
  auto GetName = [](const EntryPointStat *M) { return M->name(); };
  enumerateStatVectors([GetName, &Ret](const auto &Stats) {
    transform(Stats, std::back_inserter(Ret), GetName);
  });
  return Ret;
}

static std::string getUSR(const Decl *D) {
  llvm::SmallVector<char> Buf;
  if (index::generateUSRForDecl(D, Buf)) {
    assert(false && "This should never fail");
    return AnalysisDeclContext::getFunctionName(D);
  }
  return llvm::toStringRef(Buf).str();
}

void Registry::Snapshot::dumpAsCSV(llvm::raw_ostream &OS) const {
  auto PrintAsUnsignOpt = [&OS](std::optional<unsigned> U) {
    OS << (U.has_value() ? std::to_string(*U) : "");
  };
  auto CommaIfNeeded = [&OS](const auto &Vec1, const auto &Vec2) {
    if (!Vec1.empty() && !Vec2.empty())
      OS << ",";
  };
  auto PrintAsUnsigned = [&OS](unsigned U) { OS << U; };

  OS << '"';
  llvm::printEscapedString(getUSR(EntryPoint), OS);
  OS << "\",\"";
  OS << StatsRegistry->EscapedCPPFileName << "\",\"";
  llvm::printEscapedString(
      clang::AnalysisDeclContext::getFunctionName(EntryPoint), OS);
  OS << "\",";
  llvm::interleave(ExplicitlySetStatValues, OS, PrintAsUnsignOpt, ",");
  CommaIfNeeded(ExplicitlySetStatValues, MaxOrCountStatValues);
  llvm::interleave(MaxOrCountStatValues, OS, PrintAsUnsigned, ",");
}

void EntryPointStat::takeSnapshot(const Decl *EntryPoint) {
  auto ExplicitlySetValues = consumeExplicitlySetStats();
  auto MaxOrCounterValues = consumeMaxAndCounterStats();
  StatsRegistry->Snapshots.push_back({EntryPoint,
                                      std::move(ExplicitlySetValues),
                                      std::move(MaxOrCounterValues)});
}

void EntryPointStat::dumpStatsAsCSV(llvm::StringRef FileName) {
  std::error_code EC;
  llvm::raw_fd_ostream File(FileName, EC, llvm::sys::fs::OF_Text);
  if (EC)
    return;
  dumpStatsAsCSV(File);
}

void EntryPointStat::dumpStatsAsCSV(llvm::raw_ostream &OS) {
  OS << "USR,File,DebugName,";
  llvm::interleave(getStatNames(), OS, [&OS](const auto &a) { OS << a; }, ",");
  OS << "\n";

  std::vector<std::string> Rows;
  Rows.reserve(StatsRegistry->Snapshots.size());
  for (const auto &Snapshot : StatsRegistry->Snapshots) {
    std::string Row;
    llvm::raw_string_ostream RowOs(Row);
    Snapshot.dumpAsCSV(RowOs);
    RowOs << "\n";
    Rows.push_back(RowOs.str());
  }
  llvm::sort(Rows);
  for (const auto &Row : Rows) {
    OS << Row;
  }
}
