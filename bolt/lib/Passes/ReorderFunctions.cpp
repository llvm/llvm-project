//===- bolt/Passes/ReorderFunctions.cpp - Function reordering pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ReorderFunctions class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ReorderFunctions.h"
#include "bolt/Passes/HFSort.h"
#include "llvm/Support/CommandLine.h"
#include <fstream>

#define DEBUG_TYPE "hfsort"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;
extern cl::opt<unsigned> Verbosity;
extern cl::opt<uint32_t> RandomSeed;

extern size_t padFunction(const bolt::BinaryFunction &Function);

cl::opt<bolt::ReorderFunctions::ReorderType>
ReorderFunctions("reorder-functions",
  cl::desc("reorder and cluster functions (works only with relocations)"),
  cl::init(bolt::ReorderFunctions::RT_NONE),
  cl::values(clEnumValN(bolt::ReorderFunctions::RT_NONE,
      "none",
      "do not reorder functions"),
    clEnumValN(bolt::ReorderFunctions::RT_EXEC_COUNT,
      "exec-count",
      "order by execution count"),
    clEnumValN(bolt::ReorderFunctions::RT_HFSORT,
      "hfsort",
      "use hfsort algorithm"),
    clEnumValN(bolt::ReorderFunctions::RT_HFSORT_PLUS,
      "hfsort+",
      "use hfsort+ algorithm"),
    clEnumValN(bolt::ReorderFunctions::RT_PETTIS_HANSEN,
      "pettis-hansen",
      "use Pettis-Hansen algorithm"),
    clEnumValN(bolt::ReorderFunctions::RT_RANDOM,
      "random",
      "reorder functions randomly"),
    clEnumValN(bolt::ReorderFunctions::RT_USER,
      "user",
      "use function order specified by -function-order")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool> ReorderFunctionsUseHotSize(
    "reorder-functions-use-hot-size",
    cl::desc("use a function's hot size when doing clustering"), cl::init(true),
    cl::cat(BoltOptCategory));

static cl::opt<std::string>
FunctionOrderFile("function-order",
  cl::desc("file containing an ordered list of functions to use for function "
           "reordering"),
  cl::cat(BoltOptCategory));

static cl::opt<std::string>
GenerateFunctionOrderFile("generate-function-order",
  cl::desc("file to dump the ordered list of functions to use for function "
           "reordering"),
  cl::cat(BoltOptCategory));

static cl::opt<std::string>
LinkSectionsFile("generate-link-sections",
  cl::desc("generate a list of function sections in a format suitable for "
           "inclusion in a linker script"),
  cl::cat(BoltOptCategory));

static cl::opt<bool>
    UseEdgeCounts("use-edge-counts",
                  cl::desc("use edge count data when doing clustering"),
                  cl::init(true), cl::cat(BoltOptCategory));

static cl::opt<bool>
CgFromPerfData("cg-from-perf-data",
  cl::desc("use perf data directly when constructing the call graph"
           " for stale functions"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

static cl::opt<bool> CgIgnoreRecursiveCalls(
    "cg-ignore-recursive-calls",
    cl::desc("ignore recursive calls when constructing the call graph"),
    cl::init(true), cl::cat(BoltOptCategory));

static cl::opt<bool>
CgUseSplitHotSize("cg-use-split-hot-size",
  cl::desc("use hot/cold data on basic blocks to determine hot sizes for "
           "call graph functions"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

using NodeId = CallGraph::NodeId;
using Arc = CallGraph::Arc;
using Node = CallGraph::Node;

void ReorderFunctions::reorder(std::vector<Cluster> &&Clusters,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::vector<uint64_t> FuncAddr(Cg.numNodes()); // Just for computing stats
  uint64_t TotalSize = 0;
  uint32_t Index = 0;

  // Set order of hot functions based on clusters.
  for (const Cluster &Cluster : Clusters) {
    for (const NodeId FuncId : Cluster.targets()) {
      Cg.nodeIdToFunc(FuncId)->setIndex(Index++);
      FuncAddr[FuncId] = TotalSize;
      TotalSize += Cg.size(FuncId);
    }
  }

  // Assign valid index for functions with valid profile.
  for (auto &It : BFs) {
    BinaryFunction &BF = It.second;
    if (!BF.hasValidIndex() && BF.hasValidProfile())
      BF.setIndex(Index++);
  }

  if (opts::ReorderFunctions == RT_NONE)
    return;

  if (opts::Verbosity == 0) {
#ifndef NDEBUG
    if (!DebugFlag || !isCurrentDebugType("hfsort"))
      return;
#else
    return;
#endif
  }

  bool PrintDetailed = opts::Verbosity > 1;
#ifndef NDEBUG
  PrintDetailed |=
    (DebugFlag && isCurrentDebugType("hfsort") && opts::Verbosity > 0);
#endif
  TotalSize   = 0;
  uint64_t CurPage     = 0;
  uint64_t Hotfuncs    = 0;
  double TotalDistance = 0;
  double TotalCalls    = 0;
  double TotalCalls64B = 0;
  double TotalCalls4KB = 0;
  double TotalCalls2MB = 0;
  if (PrintDetailed)
    outs() << "BOLT-INFO: Function reordering page layout\n"
           << "BOLT-INFO: ============== page 0 ==============\n";
  for (Cluster &Cluster : Clusters) {
    if (PrintDetailed)
      outs() << format(
          "BOLT-INFO: -------- density = %.3lf (%u / %u) --------\n",
          Cluster.density(), Cluster.samples(), Cluster.size());

    for (NodeId FuncId : Cluster.targets()) {
      if (Cg.samples(FuncId) > 0) {
        Hotfuncs++;

        if (PrintDetailed)
          outs() << "BOLT-INFO: hot func " << *Cg.nodeIdToFunc(FuncId) << " ("
                 << Cg.size(FuncId) << ")\n";

        uint64_t Dist = 0;
        uint64_t Calls = 0;
        for (NodeId Dst : Cg.successors(FuncId)) {
          if (FuncId == Dst) // ignore recursive calls in stats
            continue;
          const Arc &Arc = *Cg.findArc(FuncId, Dst);
          const auto D = std::abs(FuncAddr[Arc.dst()] -
                                  (FuncAddr[FuncId] + Arc.avgCallOffset()));
          const double W = Arc.weight();
          if (D < 64 && PrintDetailed && opts::Verbosity > 2)
            outs() << "BOLT-INFO: short (" << D << "B) call:\n"
                   << "BOLT-INFO:   Src: " << *Cg.nodeIdToFunc(FuncId) << "\n"
                   << "BOLT-INFO:   Dst: " << *Cg.nodeIdToFunc(Dst) << "\n"
                   << "BOLT-INFO:   Weight = " << W << "\n"
                   << "BOLT-INFO:   AvgOffset = " << Arc.avgCallOffset() << "\n";
          Calls += W;
          if (D < 64)        TotalCalls64B += W;
          if (D < 4096)      TotalCalls4KB += W;
          if (D < (2 << 20)) TotalCalls2MB += W;
          Dist += Arc.weight() * D;
          if (PrintDetailed)
            outs() << format("BOLT-INFO: arc: %u [@%lu+%.1lf] -> %u [@%lu]: "
                             "weight = %.0lf, callDist = %f\n",
                             Arc.src(),
                             FuncAddr[Arc.src()],
                             Arc.avgCallOffset(),
                             Arc.dst(),
                             FuncAddr[Arc.dst()],
                             Arc.weight(), D);
        }
        TotalCalls += Calls;
        TotalDistance += Dist;
        TotalSize += Cg.size(FuncId);

        if (PrintDetailed) {
          outs() << format("BOLT-INFO: start = %6u : avgCallDist = %lu : ",
                           TotalSize, Calls ? Dist / Calls : 0)
                 << Cg.nodeIdToFunc(FuncId)->getPrintName() << '\n';
          const uint64_t NewPage = TotalSize / HugePageSize;
          if (NewPage != CurPage) {
            CurPage = NewPage;
            outs() << format(
                "BOLT-INFO: ============== page %u ==============\n", CurPage);
          }
        }
      }
    }
  }
  outs() << "BOLT-INFO: Function reordering stats\n"
         << format("BOLT-INFO:  Number of hot functions: %u\n"
                   "BOLT-INFO:  Number of clusters: %lu\n",
                   Hotfuncs, Clusters.size())
         << format("BOLT-INFO:  Final average call distance = %.1lf "
                   "(%.0lf / %.0lf)\n",
                   TotalCalls ? TotalDistance / TotalCalls : 0, TotalDistance,
                   TotalCalls)
         << format("BOLT-INFO:  Total Calls = %.0lf\n", TotalCalls);
  if (TotalCalls)
    outs() << format("BOLT-INFO:  Total Calls within 64B = %.0lf (%.2lf%%)\n",
                     TotalCalls64B, 100 * TotalCalls64B / TotalCalls)
           << format("BOLT-INFO:  Total Calls within 4KB = %.0lf (%.2lf%%)\n",
                     TotalCalls4KB, 100 * TotalCalls4KB / TotalCalls)
           << format("BOLT-INFO:  Total Calls within 2MB = %.0lf (%.2lf%%)\n",
                     TotalCalls2MB, 100 * TotalCalls2MB / TotalCalls);
}

namespace {

std::vector<std::string> readFunctionOrderFile() {
  std::vector<std::string> FunctionNames;
  std::ifstream FuncsFile(opts::FunctionOrderFile, std::ios::in);
  if (!FuncsFile) {
    errs() << "Ordered functions file \"" << opts::FunctionOrderFile
           << "\" can't be opened.\n";
    exit(1);
  }
  std::string FuncName;
  while (std::getline(FuncsFile, FuncName))
    FunctionNames.push_back(FuncName);
  return FunctionNames;
}

}

void ReorderFunctions::runOnFunctions(BinaryContext &BC) {
  auto &BFs = BC.getBinaryFunctions();
  if (opts::ReorderFunctions != RT_NONE &&
      opts::ReorderFunctions != RT_EXEC_COUNT &&
      opts::ReorderFunctions != RT_USER) {
    Cg = buildCallGraph(BC,
                        [](const BinaryFunction &BF) {
                          if (!BF.hasProfile())
                            return true;
                          if (BF.getState() != BinaryFunction::State::CFG)
                            return true;
                          return false;
                        },
                        opts::CgFromPerfData,
                        false, // IncludeColdCalls
                        opts::ReorderFunctionsUseHotSize,
                        opts::CgUseSplitHotSize,
                        opts::UseEdgeCounts,
                        opts::CgIgnoreRecursiveCalls);
    Cg.normalizeArcWeights();
  }

  std::vector<Cluster> Clusters;

  switch (opts::ReorderFunctions) {
  case RT_NONE:
    break;
  case RT_EXEC_COUNT:
    {
      std::vector<BinaryFunction *> SortedFunctions(BFs.size());
      uint32_t Index = 0;
      llvm::transform(BFs, SortedFunctions.begin(),
                      [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                        return &BFI.second;
                      });
      llvm::stable_sort(SortedFunctions, [&](const BinaryFunction *A,
                                             const BinaryFunction *B) {
        if (A->isIgnored())
          return false;
        const size_t PadA = opts::padFunction(*A);
        const size_t PadB = opts::padFunction(*B);
        if (!PadA || !PadB) {
          if (PadA)
            return true;
          if (PadB)
            return false;
        }
        return !A->hasProfile() &&
               (B->hasProfile() ||
                (A->getExecutionCount() > B->getExecutionCount()));
      });
      for (BinaryFunction *BF : SortedFunctions)
        if (BF->hasProfile())
          BF->setIndex(Index++);
    }
    break;
  case RT_HFSORT:
    Clusters = clusterize(Cg);
    break;
  case RT_HFSORT_PLUS:
    Clusters = hfsortPlus(Cg);
    break;
  case RT_PETTIS_HANSEN:
    Clusters = pettisAndHansen(Cg);
    break;
  case RT_RANDOM:
    std::srand(opts::RandomSeed);
    Clusters = randomClusters(Cg);
    break;
  case RT_USER:
    {
      uint32_t Index = 0;
      for (const std::string &Function : readFunctionOrderFile()) {
        std::vector<uint64_t> FuncAddrs;

        BinaryData *BD = BC.getBinaryDataByName(Function);
        if (!BD) {
          uint32_t LocalID = 1;
          while(1) {
            // If we can't find the main symbol name, look for alternates.
            const std::string FuncName =
                Function + "/" + std::to_string(LocalID);
            BD = BC.getBinaryDataByName(FuncName);
            if (BD)
              FuncAddrs.push_back(BD->getAddress());
            else
              break;
            LocalID++;
          }
        } else {
          FuncAddrs.push_back(BD->getAddress());
        }

        if (FuncAddrs.empty()) {
          errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                 << Function << ".\n";
          continue;
        }

        for (const uint64_t FuncAddr : FuncAddrs) {
          const BinaryData *FuncBD = BC.getBinaryDataAtAddress(FuncAddr);
          assert(FuncBD);

          BinaryFunction *BF = BC.getFunctionForSymbol(FuncBD->getSymbol());
          if (!BF) {
            errs() << "BOLT-WARNING: Reorder functions: can't find function for "
                   << Function << ".\n";
            break;
          }
          if (!BF->hasValidIndex())
            BF->setIndex(Index++);
          else if (opts::Verbosity > 0)
            errs() << "BOLT-WARNING: Duplicate reorder entry for " << Function
                   << ".\n";
        }
      }
    }
    break;
  }

  reorder(std::move(Clusters), BFs);

  std::unique_ptr<std::ofstream> FuncsFile;
  if (!opts::GenerateFunctionOrderFile.empty()) {
    FuncsFile = std::make_unique<std::ofstream>(opts::GenerateFunctionOrderFile,
                                                std::ios::out);
    if (!FuncsFile) {
      errs() << "BOLT-ERROR: ordered functions file "
             << opts::GenerateFunctionOrderFile << " cannot be opened\n";
      exit(1);
    }
  }

  std::unique_ptr<std::ofstream> LinkSectionsFile;
  if (!opts::LinkSectionsFile.empty()) {
    LinkSectionsFile =
        std::make_unique<std::ofstream>(opts::LinkSectionsFile, std::ios::out);
    if (!LinkSectionsFile) {
      errs() << "BOLT-ERROR: link sections file " << opts::LinkSectionsFile
             << " cannot be opened\n";
      exit(1);
    }
  }

  if (FuncsFile || LinkSectionsFile) {
    std::vector<BinaryFunction *> SortedFunctions(BFs.size());
    llvm::transform(BFs, SortedFunctions.begin(),
                    [](std::pair<const uint64_t, BinaryFunction> &BFI) {
                      return &BFI.second;
                    });

    // Sort functions by index.
    llvm::stable_sort(SortedFunctions,
                      [](const BinaryFunction *A, const BinaryFunction *B) {
                        if (A->hasValidIndex() && B->hasValidIndex())
                          return A->getIndex() < B->getIndex();
                        if (A->hasValidIndex() && !B->hasValidIndex())
                          return true;
                        if (!A->hasValidIndex() && B->hasValidIndex())
                          return false;
                        return A->getAddress() < B->getAddress();
                      });

    for (const BinaryFunction *Func : SortedFunctions) {
      if (!Func->hasValidIndex())
        break;
      if (Func->isPLTFunction())
        continue;

      if (FuncsFile)
        *FuncsFile << Func->getOneName().str() << '\n';

      if (LinkSectionsFile) {
        const char *Indent = "";
        std::vector<StringRef> AllNames = Func->getNames();
        llvm::sort(AllNames);
        for (StringRef Name : AllNames) {
          const size_t SlashPos = Name.find('/');
          if (SlashPos != std::string::npos) {
            // Avoid duplicates for local functions.
            if (Name.find('/', SlashPos + 1) != std::string::npos)
              continue;
            Name = Name.substr(0, SlashPos);
          }
          *LinkSectionsFile << Indent << ".text." << Name.str() << '\n';
          Indent = " ";
        }
      }
    }

    if (FuncsFile) {
      FuncsFile->close();
      outs() << "BOLT-INFO: dumped function order to "
             << opts::GenerateFunctionOrderFile << '\n';
    }

    if (LinkSectionsFile) {
      LinkSectionsFile->close();
      outs() << "BOLT-INFO: dumped linker section order to "
             << opts::LinkSectionsFile << '\n';
    }
  }
}

} // namespace bolt
} // namespace llvm
