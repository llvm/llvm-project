//===- bolt/Rewrite/BoltDiff.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RewriteInstance methods related to comparing one instance to another, used
// by the boltdiff tool to print a report.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/IdenticalCodeFolding.h"
#include "bolt/Profile/ProfileReaderBase.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "llvm/Support/CommandLine.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "boltdiff"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {
extern cl::OptionCategory BoltDiffCategory;
extern cl::opt<bool> NeverPrint;
extern cl::opt<bool> ICF;

static cl::opt<bool> IgnoreLTOSuffix(
    "ignore-lto-suffix",
    cl::desc("ignore lto_priv or const suffixes when matching functions"),
    cl::init(true), cl::cat(BoltDiffCategory));

static cl::opt<bool> PrintUnmapped(
    "print-unmapped",
    cl::desc("print functions of binary 2 that were not matched to any "
             "function in binary 1"),
    cl::cat(BoltDiffCategory));

static cl::opt<bool> PrintProfiledUnmapped(
    "print-profiled-unmapped",
    cl::desc("print functions that have profile in binary 1 but do not "
             "in binary 2"),
    cl::cat(BoltDiffCategory));

static cl::opt<bool> PrintDiffCFG(
    "print-diff-cfg",
    cl::desc("print the CFG of important functions that changed in "
             "binary 2"),
    cl::cat(BoltDiffCategory));

static cl::opt<bool>
    PrintDiffBBs("print-diff-bbs",
                 cl::desc("print the basic blocks showed in top differences"),
                 cl::cat(BoltDiffCategory));

static cl::opt<bool> MatchByHash(
    "match-by-hash",
    cl::desc("match functions in binary 2 to binary 1 if they have the same "
             "hash of a function in binary 1"),
    cl::cat(BoltDiffCategory));

static cl::opt<bool> IgnoreUnchanged(
    "ignore-unchanged",
    cl::desc("do not diff functions whose contents have not been changed from "
             "one binary to another"),
    cl::cat(BoltDiffCategory));

static cl::opt<unsigned> DisplayCount(
    "display-count",
    cl::desc("number of functions to display when printing the top largest "
             "differences in function activity"),
    cl::init(10), cl::cat(BoltDiffCategory));

static cl::opt<bool> NormalizeByBin1(
    "normalize-by-bin1",
    cl::desc("show execution count of functions in binary 2 as a ratio of the "
             "total samples in binary 1 - make sure both profiles have equal "
             "collection time and sampling rate for this to make sense"),
    cl::cat(BoltDiffCategory));

} // end namespace opts

namespace llvm {
namespace bolt {

namespace {

/// Helper used to print colored numbers
void printColoredPercentage(double Perc) {
  if (outs().has_colors() && Perc > 0.0)
    outs().changeColor(raw_ostream::RED);
  else if (outs().has_colors() && Perc < 0.0)
    outs().changeColor(raw_ostream::GREEN);
  else if (outs().has_colors())
    outs().changeColor(raw_ostream::YELLOW);
  outs() << format("%.2f", Perc) << "%";
  if (outs().has_colors())
    outs().resetColor();
}

void setLightColor() {
  if (opts::PrintDiffBBs && outs().has_colors())
    outs().changeColor(raw_ostream::CYAN);
}

void setTitleColor() {
  if (outs().has_colors())
    outs().changeColor(raw_ostream::WHITE, /*Bold=*/true);
}

void setRegularColor() {
  if (outs().has_colors())
    outs().resetColor();
}

} // end anonymous namespace

/// Perform the comparison between two binaries with profiling information
class RewriteInstanceDiff {
  typedef std::tuple<const BinaryBasicBlock *, const BinaryBasicBlock *, double>
      EdgeTy;

  RewriteInstance &RI1;
  RewriteInstance &RI2;

  // The map of functions keyed by functions in binary 2, providing its
  // corresponding function in binary 1
  std::map<const BinaryFunction *, const BinaryFunction *> FuncMap;

  // The map of basic blocks correspondence, analogue to FuncMap for BBs,
  // sorted by score difference
  std::map<const BinaryBasicBlock *, const BinaryBasicBlock *> BBMap;

  // The map of edge correspondence
  std::map<double, std::pair<EdgeTy, EdgeTy>> EdgeMap;

  // Maps all known basic blocks back to their parent function
  std::map<const BinaryBasicBlock *, const BinaryFunction *> BBToFuncMap;

  // Accounting which functions were matched
  std::set<const BinaryFunction *> Bin1MappedFuncs;
  std::set<const BinaryFunction *> Bin2MappedFuncs;

  // Structures for our 3 matching strategies: by name, by hash and by lto name,
  // from the strongest to the weakest bind between two functions
  StringMap<const BinaryFunction *> NameLookup;
  DenseMap<size_t, const BinaryFunction *> HashLookup;
  StringMap<const BinaryFunction *> LTONameLookup1;
  StringMap<const BinaryFunction *> LTONameLookup2;

  // Score maps used to order and find hottest functions
  std::multimap<double, const BinaryFunction *> LargestBin1;
  std::multimap<double, const BinaryFunction *> LargestBin2;

  // Map multiple functions in the same LTO bucket to a single parent function
  // representing all functions sharing the same prefix
  std::map<const BinaryFunction *, const BinaryFunction *> LTOMap1;
  std::map<const BinaryFunction *, const BinaryFunction *> LTOMap2;
  std::map<const BinaryFunction *, double> LTOAggregatedScore1;
  std::map<const BinaryFunction *, double> LTOAggregatedScore2;

  // Map scores in bin2 and 1 keyed by a binary 2 function - post-matching
  DenseMap<const BinaryFunction *, std::pair<double, double>> ScoreMap;

  double getNormalizedScore(const BinaryFunction &Function,
                            const RewriteInstance &Ctx) {
    if (!opts::NormalizeByBin1)
      return static_cast<double>(Function.getFunctionScore()) /
             Ctx.getTotalScore();
    return static_cast<double>(Function.getFunctionScore()) /
           RI1.getTotalScore();
  }

  double getNormalizedScore(const BinaryBasicBlock &BB,
                            const RewriteInstance &Ctx) {
    if (!opts::NormalizeByBin1)
      return static_cast<double>(BB.getKnownExecutionCount()) /
             Ctx.getTotalScore();
    return static_cast<double>(BB.getKnownExecutionCount()) /
           RI1.getTotalScore();
  }

  double getNormalizedScore(BinaryBasicBlock::branch_info_iterator BIIter,
                            const RewriteInstance &Ctx) {
    double Score =
        BIIter->Count == BinaryBasicBlock::COUNT_NO_PROFILE ? 0 : BIIter->Count;
    if (!opts::NormalizeByBin1)
      return Score / Ctx.getTotalScore();
    return Score / RI1.getTotalScore();
  }

  /// Initialize data structures used for function lookup in binary 1, used
  /// later when matching functions in binary 2 to corresponding functions
  /// in binary 1
  void buildLookupMaps() {
    for (const auto &BFI : RI1.BC->getBinaryFunctions()) {
      StringRef LTOName;
      const BinaryFunction &Function = BFI.second;
      const double Score = getNormalizedScore(Function, RI1);
      LargestBin1.insert(std::make_pair<>(Score, &Function));
      for (const StringRef &Name : Function.getNames()) {
        if (Optional<StringRef> OptionalLTOName = getLTOCommonName(Name))
          LTOName = *OptionalLTOName;
        NameLookup[Name] = &Function;
      }
      if (opts::MatchByHash && Function.hasCFG())
        HashLookup[Function.computeHash(/*UseDFS=*/true)] = &Function;
      if (opts::IgnoreLTOSuffix && !LTOName.empty()) {
        if (!LTONameLookup1.count(LTOName))
          LTONameLookup1[LTOName] = &Function;
        LTOMap1[&Function] = LTONameLookup1[LTOName];
      }
    }

    // Compute LTONameLookup2 and LargestBin2
    for (const auto &BFI : RI2.BC->getBinaryFunctions()) {
      StringRef LTOName;
      const BinaryFunction &Function = BFI.second;
      const double Score = getNormalizedScore(Function, RI2);
      LargestBin2.insert(std::make_pair<>(Score, &Function));
      for (const StringRef &Name : Function.getNames()) {
        if (Optional<StringRef> OptionalLTOName = getLTOCommonName(Name))
          LTOName = *OptionalLTOName;
      }
      if (opts::IgnoreLTOSuffix && !LTOName.empty()) {
        if (!LTONameLookup2.count(LTOName))
          LTONameLookup2[LTOName] = &Function;
        LTOMap2[&Function] = LTONameLookup2[LTOName];
      }
    }
  }

  /// Match functions in binary 2 with functions in binary 1
  void matchFunctions() {
    outs() << "BOLT-DIFF: Mapping functions in Binary2 to Binary1\n";
    uint64_t BothHaveProfile = 0ull;
    std::set<const BinaryFunction *> Bin1ProfiledMapped;

    for (const auto &BFI2 : RI2.BC->getBinaryFunctions()) {
      const BinaryFunction &Function2 = BFI2.second;
      StringRef LTOName;
      bool Match = false;
      for (const StringRef &Name : Function2.getNames()) {
        auto Iter = NameLookup.find(Name);
        if (Optional<StringRef> OptionalLTOName = getLTOCommonName(Name))
          LTOName = *OptionalLTOName;
        if (Iter == NameLookup.end())
          continue;
        FuncMap.insert(std::make_pair<>(&Function2, Iter->second));
        Bin1MappedFuncs.insert(Iter->second);
        Bin2MappedFuncs.insert(&Function2);
        if (Function2.hasValidProfile() && Iter->second->hasValidProfile()) {
          ++BothHaveProfile;
          Bin1ProfiledMapped.insert(Iter->second);
        }
        Match = true;
        break;
      }
      if (Match || !Function2.hasCFG())
        continue;
      auto Iter = HashLookup.find(Function2.computeHash(/*UseDFS*/ true));
      if (Iter != HashLookup.end()) {
        FuncMap.insert(std::make_pair<>(&Function2, Iter->second));
        Bin1MappedFuncs.insert(Iter->second);
        Bin2MappedFuncs.insert(&Function2);
        if (Function2.hasValidProfile() && Iter->second->hasValidProfile()) {
          ++BothHaveProfile;
          Bin1ProfiledMapped.insert(Iter->second);
        }
        continue;
      }
      if (LTOName.empty())
        continue;
      auto LTOIter = LTONameLookup1.find(LTOName);
      if (LTOIter != LTONameLookup1.end()) {
        FuncMap.insert(std::make_pair<>(&Function2, LTOIter->second));
        Bin1MappedFuncs.insert(LTOIter->second);
        Bin2MappedFuncs.insert(&Function2);
        if (Function2.hasValidProfile() && LTOIter->second->hasValidProfile()) {
          ++BothHaveProfile;
          Bin1ProfiledMapped.insert(LTOIter->second);
        }
      }
    }
    PrintProgramStats PPS(opts::NeverPrint);
    outs() << "* BOLT-DIFF: Starting print program stats pass for binary 1\n";
    PPS.runOnFunctions(*RI1.BC);
    outs() << "* BOLT-DIFF: Starting print program stats pass for binary 2\n";
    PPS.runOnFunctions(*RI2.BC);
    outs() << "=====\n";
    outs() << "Inputs share " << BothHaveProfile
           << " functions with valid profile.\n";
    if (opts::PrintProfiledUnmapped) {
      outs() << "\nFunctions in profile 1 that are missing in the profile 2:\n";
      std::vector<const BinaryFunction *> Unmapped;
      for (const auto &BFI : RI1.BC->getBinaryFunctions()) {
        const BinaryFunction &Function = BFI.second;
        if (!Function.hasValidProfile() || Bin1ProfiledMapped.count(&Function))
          continue;
        Unmapped.emplace_back(&Function);
      }
      llvm::sort(Unmapped,
                 [&](const BinaryFunction *A, const BinaryFunction *B) {
                   return A->getFunctionScore() > B->getFunctionScore();
                 });
      for (const BinaryFunction *Function : Unmapped) {
        outs() << Function->getPrintName() << " : ";
        outs() << Function->getFunctionScore() << "\n";
      }
      outs() << "=====\n";
    }
  }

  /// Check if opcodes in BB1 match those in BB2
  bool compareBBs(const BinaryBasicBlock &BB1,
                  const BinaryBasicBlock &BB2) const {
    auto Iter1 = BB1.begin();
    auto Iter2 = BB2.begin();
    if ((Iter1 == BB1.end() && Iter2 != BB2.end()) ||
        (Iter1 != BB1.end() && Iter2 == BB2.end()))
      return false;

    while (Iter1 != BB1.end()) {
      if (Iter2 == BB2.end() || Iter1->getOpcode() != Iter2->getOpcode())
        return false;

      ++Iter1;
      ++Iter2;
    }

    if (Iter2 != BB2.end())
      return false;
    return true;
  }

  /// For a function in binary 2 that matched one in binary 1, now match each
  /// individual basic block in it to its corresponding blocks in binary 1.
  /// Also match each edge in binary 2 to the corresponding ones in binary 1.
  void matchBasicBlocks() {
    for (const auto &MapEntry : FuncMap) {
      const BinaryFunction *const &Func1 = MapEntry.second;
      const BinaryFunction *const &Func2 = MapEntry.first;

      auto Iter1 = Func1->layout_begin();
      auto Iter2 = Func2->layout_begin();

      bool Match = true;
      std::map<const BinaryBasicBlock *, const BinaryBasicBlock *> Map;
      std::map<double, std::pair<EdgeTy, EdgeTy>> EMap;
      while (Iter1 != Func1->layout_end()) {
        if (Iter2 == Func2->layout_end()) {
          Match = false;
          break;
        }
        if (!compareBBs(**Iter1, **Iter2)) {
          Match = false;
          break;
        }
        Map.insert(std::make_pair<>(*Iter2, *Iter1));

        auto SuccIter1 = (*Iter1)->succ_begin();
        auto SuccIter2 = (*Iter2)->succ_begin();
        auto BIIter1 = (*Iter1)->branch_info_begin();
        auto BIIter2 = (*Iter2)->branch_info_begin();
        while (SuccIter1 != (*Iter1)->succ_end()) {
          if (SuccIter2 == (*Iter2)->succ_end()) {
            Match = false;
            break;
          }
          const double ScoreEdge1 = getNormalizedScore(BIIter1, RI1);
          const double ScoreEdge2 = getNormalizedScore(BIIter2, RI2);
          EMap.insert(std::make_pair<>(
              std::abs(ScoreEdge2 - ScoreEdge1),
              std::make_pair<>(
                  std::make_tuple<>(*Iter2, *SuccIter2, ScoreEdge2),
                  std::make_tuple<>(*Iter1, *SuccIter1, ScoreEdge1))));

          ++SuccIter1;
          ++SuccIter2;
          ++BIIter1;
          ++BIIter2;
        }
        if (SuccIter2 != (*Iter2)->succ_end())
          Match = false;
        if (!Match)
          break;

        BBToFuncMap[*Iter1] = Func1;
        BBToFuncMap[*Iter2] = Func2;
        ++Iter1;
        ++Iter2;
      }
      if (!Match || Iter2 != Func2->layout_end())
        continue;

      BBMap.insert(Map.begin(), Map.end());
      EdgeMap.insert(EMap.begin(), EMap.end());
    }
  }

  /// Print the largest differences in basic block performance from binary 1
  /// to binary 2
  void reportHottestBBDiffs() {
    std::map<double, const BinaryBasicBlock *> LargestDiffs;
    for (const auto &MapEntry : BBMap) {
      const BinaryBasicBlock *BB2 = MapEntry.first;
      const BinaryBasicBlock *BB1 = MapEntry.second;
      LargestDiffs.insert(
          std::make_pair<>(std::abs(getNormalizedScore(*BB2, RI2) -
                                    getNormalizedScore(*BB1, RI1)),
                           BB2));
    }

    unsigned Printed = 0;
    setTitleColor();
    outs()
        << "\nTop " << opts::DisplayCount
        << " largest differences in basic block performance bin 2 -> bin 1:\n";
    outs() << "=========================================================\n";
    setRegularColor();
    outs() << " * Functions with different contents do not appear here\n\n";
    for (auto I = LargestDiffs.rbegin(), E = LargestDiffs.rend(); I != E; ++I) {
      const BinaryBasicBlock *BB2 = I->second;
      const double Score2 = getNormalizedScore(*BB2, RI2);
      const double Score1 = getNormalizedScore(*BBMap[BB2], RI1);
      outs() << "BB " << BB2->getName() << " from "
             << BBToFuncMap[BB2]->getDemangledName()
             << "\n\tScore bin1 = " << format("%.4f", Score1 * 100.0)
             << "%\n\tScore bin2 = " << format("%.4f", Score2 * 100.0);
      outs() << "%\t(Difference: ";
      printColoredPercentage((Score2 - Score1) * 100.0);
      outs() << ")\n";
      if (opts::PrintDiffBBs) {
        setLightColor();
        BB2->dump();
        setRegularColor();
      }
      if (Printed++ == opts::DisplayCount)
        break;
    }
  }

  /// Print the largest differences in edge counts from one binary to another
  void reportHottestEdgeDiffs() {
    unsigned Printed = 0;
    setTitleColor();
    outs() << "\nTop " << opts::DisplayCount
           << " largest differences in edge hotness bin 2 -> bin 1:\n";
    outs() << "=========================================================\n";
    setRegularColor();
    outs() << " * Functions with different contents do not appear here\n";
    for (auto I = EdgeMap.rbegin(), E = EdgeMap.rend(); I != E; ++I) {
      std::tuple<const BinaryBasicBlock *, const BinaryBasicBlock *, double>
          &Edge2 = I->second.first;
      std::tuple<const BinaryBasicBlock *, const BinaryBasicBlock *, double>
          &Edge1 = I->second.second;
      const double Score2 = std::get<2>(Edge2);
      const double Score1 = std::get<2>(Edge1);
      outs() << "Edge (" << std::get<0>(Edge2)->getName() << " -> "
             << std::get<1>(Edge2)->getName() << ") in "
             << BBToFuncMap[std::get<0>(Edge2)]->getDemangledName()
             << "\n\tScore bin1 = " << format("%.4f", Score1 * 100.0)
             << "%\n\tScore bin2 = " << format("%.4f", Score2 * 100.0);
      outs() << "%\t(Difference: ";
      printColoredPercentage((Score2 - Score1) * 100.0);
      outs() << ")\n";
      if (opts::PrintDiffBBs) {
        setLightColor();
        std::get<0>(Edge2)->dump();
        std::get<1>(Edge2)->dump();
        setRegularColor();
      }
      if (Printed++ == opts::DisplayCount)
        break;
    }
  }

  /// For LTO functions sharing the same prefix (for example, func1.lto_priv.1
  /// and func1.lto_priv.2 share the func1.lto_priv prefix), compute aggregated
  /// scores for them. This is used to avoid reporting all LTO functions as
  /// having a large difference in performance because hotness shifted from
  /// LTO variant 1 to variant 2, even though they represent the same function.
  void computeAggregatedLTOScore() {
    for (const auto &BFI : RI1.BC->getBinaryFunctions()) {
      const BinaryFunction &Function = BFI.second;
      double Score = getNormalizedScore(Function, RI1);
      auto Iter = LTOMap1.find(&Function);
      if (Iter == LTOMap1.end())
        continue;
      LTOAggregatedScore1[Iter->second] += Score;
    }

    double UnmappedScore = 0;
    for (const auto &BFI : RI2.BC->getBinaryFunctions()) {
      const BinaryFunction &Function = BFI.second;
      bool Matched = FuncMap.find(&Function) != FuncMap.end();
      double Score = getNormalizedScore(Function, RI2);
      auto Iter = LTOMap2.find(&Function);
      if (Iter == LTOMap2.end()) {
        if (!Matched)
          UnmappedScore += Score;
        continue;
      }
      LTOAggregatedScore2[Iter->second] += Score;
      if (FuncMap.find(Iter->second) == FuncMap.end())
        UnmappedScore += Score;
    }
    int64_t Unmapped =
        RI2.BC->getBinaryFunctions().size() - Bin2MappedFuncs.size();
    outs() << "BOLT-DIFF: " << Unmapped
           << " functions in Binary2 have no correspondence to any other "
              "function in Binary1.\n";

    // Print the hotness score of functions in binary 2 that were not matched
    // to any function in binary 1
    outs() << "BOLT-DIFF: These unmapped functions in Binary2 represent "
           << format("%.2f", UnmappedScore * 100.0) << "% of execution.\n";
  }

  /// Print the largest hotness differences from binary 2 to binary 1
  void reportHottestFuncDiffs() {
    std::multimap<double, decltype(FuncMap)::value_type> LargestDiffs;
    for (const auto &MapEntry : FuncMap) {
      const BinaryFunction *const &Func1 = MapEntry.second;
      const BinaryFunction *const &Func2 = MapEntry.first;
      double Score1 = getNormalizedScore(*Func1, RI1);
      auto Iter1 = LTOMap1.find(Func1);
      if (Iter1 != LTOMap1.end())
        Score1 = LTOAggregatedScore1[Iter1->second];
      double Score2 = getNormalizedScore(*Func2, RI2);
      auto Iter2 = LTOMap2.find(Func2);
      if (Iter2 != LTOMap2.end())
        Score2 = LTOAggregatedScore2[Iter2->second];
      if (Score1 == 0.0 || Score2 == 0.0)
        continue;
      LargestDiffs.insert(
          std::make_pair<>(std::abs(Score1 - Score2), MapEntry));
      ScoreMap[Func2] = std::make_pair<>(Score1, Score2);
    }

    unsigned Printed = 0;
    setTitleColor();
    outs() << "\nTop " << opts::DisplayCount
           << " largest differences in performance bin 2 -> bin 1:\n";
    outs() << "=========================================================\n";
    setRegularColor();
    for (auto I = LargestDiffs.rbegin(), E = LargestDiffs.rend(); I != E; ++I) {
      const std::pair<const BinaryFunction *const, const BinaryFunction *>
          &MapEntry = I->second;
      if (opts::IgnoreUnchanged &&
          MapEntry.second->computeHash(/*UseDFS=*/true) ==
              MapEntry.first->computeHash(/*UseDFS=*/true))
        continue;
      const std::pair<double, double> &Scores = ScoreMap[MapEntry.first];
      outs() << "Function " << MapEntry.first->getDemangledName();
      if (MapEntry.first->getDemangledName() !=
          MapEntry.second->getDemangledName())
        outs() << "\nmatched  " << MapEntry.second->getDemangledName();
      outs() << "\n\tScore bin1 = " << format("%.2f", Scores.first * 100.0)
             << "%\n\tScore bin2 = " << format("%.2f", Scores.second * 100.0)
             << "%\t(Difference: ";
      printColoredPercentage((Scores.second - Scores.first) * 100.0);
      outs() << ")";
      if (MapEntry.second->computeHash(/*UseDFS=*/true) !=
          MapEntry.first->computeHash(/*UseDFS=*/true)) {
        outs() << "\t[Functions have different contents]";
        if (opts::PrintDiffCFG) {
          outs() << "\n *** CFG for function in binary 1:\n";
          setLightColor();
          MapEntry.second->dump();
          setRegularColor();
          outs() << "\n *** CFG for function in binary 2:\n";
          setLightColor();
          MapEntry.first->dump();
          setRegularColor();
        }
      }
      outs() << "\n";
      if (Printed++ == opts::DisplayCount)
        break;
    }
  }

  /// Print hottest functions from each binary
  void reportHottestFuncs() {
    unsigned Printed = 0;
    setTitleColor();
    outs() << "\nTop " << opts::DisplayCount
           << " hottest functions in binary 2:\n";
    outs() << "=====================================\n";
    setRegularColor();
    for (auto I = LargestBin2.rbegin(), E = LargestBin2.rend(); I != E; ++I) {
      const std::pair<const double, const BinaryFunction *> &MapEntry = *I;
      outs() << "Function " << MapEntry.second->getDemangledName() << "\n";
      auto Iter = ScoreMap.find(MapEntry.second);
      if (Iter != ScoreMap.end())
        outs() << "\tScore bin1 = "
               << format("%.2f", Iter->second.first * 100.0) << "%\n";
      outs() << "\tScore bin2 = " << format("%.2f", MapEntry.first * 100.0)
             << "%\n";
      if (Printed++ == opts::DisplayCount)
        break;
    }

    Printed = 0;
    setTitleColor();
    outs() << "\nTop " << opts::DisplayCount
           << " hottest functions in binary 1:\n";
    outs() << "=====================================\n";
    setRegularColor();
    for (auto I = LargestBin1.rbegin(), E = LargestBin1.rend(); I != E; ++I) {
      const std::pair<const double, const BinaryFunction *> &MapEntry = *I;
      outs() << "Function " << MapEntry.second->getDemangledName()
             << "\n\tScore bin1 = " << format("%.2f", MapEntry.first * 100.0)
             << "%\n";
      if (Printed++ == opts::DisplayCount)
        break;
    }
  }

  /// Print functions in binary 2 that did not match anything in binary 1.
  /// Unfortunately, in an LTO build, even a small change can lead to several
  /// LTO variants being unmapped, corresponding to local functions that never
  /// appear in one of the binaries because they were previously inlined.
  void reportUnmapped() {
    outs() << "List of functions from binary 2 that were not matched with any "
           << "function in binary 1:\n";
    for (const auto &BFI2 : RI2.BC->getBinaryFunctions()) {
      const BinaryFunction &Function2 = BFI2.second;
      if (Bin2MappedFuncs.count(&Function2))
        continue;
      outs() << Function2.getPrintName() << "\n";
    }
  }

public:
  /// Main entry point: coordinate all tasks necessary to compare two binaries
  void compareAndReport() {
    buildLookupMaps();
    matchFunctions();
    if (opts::IgnoreLTOSuffix)
      computeAggregatedLTOScore();
    matchBasicBlocks();
    reportHottestFuncDiffs();
    reportHottestBBDiffs();
    reportHottestEdgeDiffs();
    reportHottestFuncs();
    if (!opts::PrintUnmapped)
      return;
    reportUnmapped();
  }

  RewriteInstanceDiff(RewriteInstance &RI1, RewriteInstance &RI2)
      : RI1(RI1), RI2(RI2) {
    compareAndReport();
  }

};

} // end nampespace bolt
} // end namespace llvm

void RewriteInstance::compare(RewriteInstance &RI2) {
  outs() << "BOLT-DIFF: ======== Binary1 vs. Binary2 ========\n";
  outs() << "Trace for binary 1 has " << this->getTotalScore()
         << " instructions executed.\n";
  outs() << "Trace for binary 2 has " << RI2.getTotalScore()
         << " instructions executed.\n";
  if (opts::NormalizeByBin1) {
    double Diff2to1 =
        static_cast<double>(RI2.getTotalScore() - this->getTotalScore()) /
        this->getTotalScore();
    outs() << "Binary2 change in score with respect to Binary1: ";
    printColoredPercentage(Diff2to1 * 100.0);
    outs() << "\n";
  }

  if (!this->getTotalScore() || !RI2.getTotalScore()) {
    outs() << "BOLT-DIFF: Both binaries must have recorded activity in known "
              "functions.\n";
    return;
  }

  // Pre-pass ICF
  if (opts::ICF) {
    IdenticalCodeFolding ICF(opts::NeverPrint);
    outs() << "BOLT-DIFF: Starting ICF pass for binary 1";
    ICF.runOnFunctions(*BC);
    outs() << "BOLT-DIFF: Starting ICF pass for binary 2";
    ICF.runOnFunctions(*RI2.BC);
  }

  RewriteInstanceDiff RID(*this, RI2);
}
