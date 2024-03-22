//===- bolt/Rewrite/BinaryPassManager.cpp - Binary-level pass manager -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Passes/ADRRelaxationPass.h"
#include "bolt/Passes/Aligner.h"
#include "bolt/Passes/AllocCombiner.h"
#include "bolt/Passes/AsmDump.h"
#include "bolt/Passes/CMOVConversion.h"
#include "bolt/Passes/FixRISCVCallsPass.h"
#include "bolt/Passes/FixRelaxationPass.h"
#include "bolt/Passes/FrameOptimizer.h"
#include "bolt/Passes/Hugify.h"
#include "bolt/Passes/IdenticalCodeFolding.h"
#include "bolt/Passes/IndirectCallPromotion.h"
#include "bolt/Passes/Inliner.h"
#include "bolt/Passes/Instrumentation.h"
#include "bolt/Passes/JTFootprintReduction.h"
#include "bolt/Passes/LongJmp.h"
#include "bolt/Passes/LoopInversionPass.h"
#include "bolt/Passes/PLTCall.h"
#include "bolt/Passes/PatchEntries.h"
#include "bolt/Passes/RegReAssign.h"
#include "bolt/Passes/ReorderData.h"
#include "bolt/Passes/ReorderFunctions.h"
#include "bolt/Passes/RetpolineInsertion.h"
#include "bolt/Passes/SplitFunctions.h"
#include "bolt/Passes/StokeInfo.h"
#include "bolt/Passes/TailDuplication.h"
#include "bolt/Passes/ThreeWayBranch.h"
#include "bolt/Passes/ValidateInternalCalls.h"
#include "bolt/Passes/ValidateMemRefs.h"
#include "bolt/Passes/VeneerElimination.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <numeric>

using namespace llvm;

namespace opts {

extern cl::opt<bool> PrintAll;
extern cl::opt<bool> PrintDynoStats;
extern cl::opt<bool> DumpDotAll;
extern cl::opt<std::string> AsmDump;
extern cl::opt<bolt::PLTCall::OptType> PLT;

static cl::opt<bool>
DynoStatsAll("dyno-stats-all",
  cl::desc("print dyno stats after each stage"),
  cl::ZeroOrMore, cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
    EliminateUnreachable("eliminate-unreachable",
                         cl::desc("eliminate unreachable code"), cl::init(true),
                         cl::cat(BoltOptCategory));

cl::opt<bool> ICF("icf", cl::desc("fold functions with identical code"),
                  cl::cat(BoltOptCategory));

static cl::opt<bool> JTFootprintReductionFlag(
    "jt-footprint-reduction",
    cl::desc("make jump tables size smaller at the cost of using more "
             "instructions at jump sites"),
    cl::cat(BoltOptCategory));

cl::opt<bool>
    KeepNops("keep-nops",
             cl::desc("keep no-op instructions. By default they are removed."),
             cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool> NeverPrint("never-print", cl::desc("never print"),
                         cl::ReallyHidden, cl::cat(BoltOptCategory));

cl::opt<bool>
PrintAfterBranchFixup("print-after-branch-fixup",
  cl::desc("print function after fixing local branches"),
  cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
PrintAfterLowering("print-after-lowering",
  cl::desc("print function after instruction lowering"),
  cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool>
PrintFinalized("print-finalized",
  cl::desc("print function after CFG is finalized"),
  cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintFOP("print-fop",
             cl::desc("print functions after frame optimizer pass"), cl::Hidden,
             cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintICF("print-icf", cl::desc("print functions after ICF optimization"),
             cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintICP("print-icp",
             cl::desc("print functions after indirect call promotion"),
             cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintInline("print-inline",
                cl::desc("print functions after inlining optimization"),
                cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> PrintJTFootprintReduction(
    "print-after-jt-footprint-reduction",
    cl::desc("print function after jt-footprint-reduction pass"), cl::Hidden,
    cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintLongJmp("print-longjmp",
                 cl::desc("print functions after longjmp pass"), cl::Hidden,
                 cl::cat(BoltOptCategory));

cl::opt<bool>
    PrintNormalized("print-normalized",
                    cl::desc("print functions after CFG is normalized"),
                    cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool> PrintOptimizeBodyless(
    "print-optimize-bodyless",
    cl::desc("print functions after bodyless optimization"), cl::Hidden,
    cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintPeepholes("print-peepholes",
                   cl::desc("print functions after peephole optimization"),
                   cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintPLT("print-plt", cl::desc("print functions after PLT optimization"),
             cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintProfileStats("print-profile-stats",
                      cl::desc("print profile quality/bias analysis"),
                      cl::cat(BoltCategory));

static cl::opt<bool>
    PrintRegReAssign("print-regreassign",
                     cl::desc("print functions after regreassign pass"),
                     cl::Hidden, cl::cat(BoltOptCategory));

cl::opt<bool>
    PrintReordered("print-reordered",
                   cl::desc("print functions after layout optimization"),
                   cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintReorderedFunctions("print-reordered-functions",
                            cl::desc("print functions after clustering"),
                            cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> PrintRetpolineInsertion(
    "print-retpoline-insertion",
    cl::desc("print functions after retpoline insertion pass"), cl::Hidden,
    cl::cat(BoltCategory));

static cl::opt<bool> PrintSCTC(
    "print-sctc",
    cl::desc("print functions after conditional tail call simplification"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> PrintSimplifyROLoads(
    "print-simplify-rodata-loads",
    cl::desc("print functions after simplification of RO data loads"),
    cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintSplit("print-split", cl::desc("print functions after code splitting"),
               cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintStoke("print-stoke", cl::desc("print functions after stoke analysis"),
               cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintFixRelaxations("print-fix-relaxations",
                        cl::desc("print functions after fix relaxations pass"),
                        cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintFixRISCVCalls("print-fix-riscv-calls",
                       cl::desc("print functions after fix RISCV calls pass"),
                       cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> PrintVeneerElimination(
    "print-veneer-elimination",
    cl::desc("print functions after veneer elimination pass"), cl::Hidden,
    cl::cat(BoltOptCategory));

static cl::opt<bool>
    PrintUCE("print-uce",
             cl::desc("print functions after unreachable code elimination"),
             cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> RegReAssign(
    "reg-reassign",
    cl::desc(
        "reassign registers so as to avoid using REX prefixes in hot code"),
    cl::cat(BoltOptCategory));

static cl::opt<bool> SimplifyConditionalTailCalls(
    "simplify-conditional-tail-calls",
    cl::desc("simplify conditional tail calls by removing unnecessary jumps"),
    cl::init(true), cl::cat(BoltOptCategory));

static cl::opt<bool> SimplifyRODataLoads(
    "simplify-rodata-loads",
    cl::desc("simplify loads from read-only sections by replacing the memory "
             "operand with the constant found in the corresponding section"),
    cl::cat(BoltOptCategory));

static cl::list<std::string>
SpecializeMemcpy1("memcpy1-spec",
  cl::desc("list of functions with call sites for which to specialize memcpy() "
           "for size 1"),
  cl::value_desc("func1,func2:cs1:cs2,func3:cs1,..."),
  cl::ZeroOrMore, cl::cat(BoltOptCategory));

static cl::opt<bool> Stoke("stoke", cl::desc("turn on the stoke analysis"),
                           cl::cat(BoltOptCategory));

static cl::opt<bool> StringOps(
    "inline-memcpy",
    cl::desc("inline memcpy using 'rep movsb' instruction (X86-only)"),
    cl::cat(BoltOptCategory));

static cl::opt<bool> StripRepRet(
    "strip-rep-ret",
    cl::desc("strip 'repz' prefix from 'repz retq' sequence (on by default)"),
    cl::init(true), cl::cat(BoltOptCategory));

static cl::opt<bool> VerifyCFG("verify-cfg",
                               cl::desc("verify the CFG after every pass"),
                               cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<bool> ThreeWayBranchFlag("three-way-branch",
                                        cl::desc("reorder three way branches"),
                                        cl::ReallyHidden,
                                        cl::cat(BoltOptCategory));

static cl::opt<bool> CMOVConversionFlag("cmov-conversion",
                                        cl::desc("fold jcc+mov into cmov"),
                                        cl::ReallyHidden,
                                        cl::cat(BoltOptCategory));

} // namespace opts

namespace llvm {
namespace bolt {

using namespace opts;

const char BinaryFunctionPassManager::TimerGroupName[] = "passman";
const char BinaryFunctionPassManager::TimerGroupDesc[] =
    "Binary Function Pass Manager";

Error BinaryFunctionPassManager::runPasses() {
  auto &BFs = BC.getBinaryFunctions();
  for (size_t PassIdx = 0; PassIdx < Passes.size(); PassIdx++) {
    const std::pair<const bool, std::unique_ptr<BinaryFunctionPass>>
        &OptPassPair = Passes[PassIdx];
    if (!OptPassPair.first)
      continue;

    const std::unique_ptr<BinaryFunctionPass> &Pass = OptPassPair.second;
    std::string PassIdName =
        formatv("{0:2}_{1}", PassIdx, Pass->getName()).str();

    if (opts::Verbosity > 0)
      BC.outs() << "BOLT-INFO: Starting pass: " << Pass->getName() << "\n";

    NamedRegionTimer T(Pass->getName(), Pass->getName(), TimerGroupName,
                       TimerGroupDesc, TimeOpts);

    Error E = Error::success();
    callWithDynoStats(
        BC.outs(),
        [this, &E, &Pass] {
          E = joinErrors(std::move(E), Pass->runOnFunctions(BC));
        },
        BFs, Pass->getName(), opts::DynoStatsAll, BC.isAArch64());
    if (E)
      return Error(std::move(E));

    if (opts::VerifyCFG &&
        !std::accumulate(
            BFs.begin(), BFs.end(), true,
            [](const bool Valid,
               const std::pair<const uint64_t, BinaryFunction> &It) {
              return Valid && It.second.validateCFG();
            })) {
      return createFatalBOLTError(
          Twine("BOLT-ERROR: Invalid CFG detected after pass ") +
          Twine(Pass->getName()) + Twine("\n"));
    }

    if (opts::Verbosity > 0)
      BC.outs() << "BOLT-INFO: Finished pass: " << Pass->getName() << "\n";

    if (!opts::PrintAll && !opts::DumpDotAll && !Pass->printPass())
      continue;

    const std::string Message = std::string("after ") + Pass->getName();

    for (auto &It : BFs) {
      BinaryFunction &Function = It.second;

      if (!Pass->shouldPrint(Function))
        continue;

      Function.print(BC.outs(), Message);

      if (opts::DumpDotAll)
        Function.dumpGraphForPass(PassIdName);
    }
  }
  return Error::success();
}

Error BinaryFunctionPassManager::runAllPasses(BinaryContext &BC) {
  BinaryFunctionPassManager Manager(BC);

  const DynoStats InitialDynoStats =
      getDynoStats(BC.getBinaryFunctions(), BC.isAArch64());

  Manager.registerPass(std::make_unique<AsmDumpPass>(),
                       opts::AsmDump.getNumOccurrences());

  if (BC.isAArch64()) {
    Manager.registerPass(std::make_unique<FixRelaxations>(PrintFixRelaxations));

    Manager.registerPass(
        std::make_unique<VeneerElimination>(PrintVeneerElimination));
  }

  if (BC.isRISCV()) {
    Manager.registerPass(
        std::make_unique<FixRISCVCallsPass>(PrintFixRISCVCalls));
  }

  // Here we manage dependencies/order manually, since passes are run in the
  // order they're registered.

  // Run this pass first to use stats for the original functions.
  Manager.registerPass(std::make_unique<PrintProgramStats>(NeverPrint));

  if (opts::PrintProfileStats)
    Manager.registerPass(std::make_unique<PrintProfileStats>(NeverPrint));

  Manager.registerPass(std::make_unique<ValidateInternalCalls>(NeverPrint));

  Manager.registerPass(std::make_unique<ValidateMemRefs>(NeverPrint));

  if (opts::Instrument)
    Manager.registerPass(std::make_unique<Instrumentation>(NeverPrint));
  else if (opts::Hugify)
    Manager.registerPass(std::make_unique<HugePage>(NeverPrint));

  Manager.registerPass(std::make_unique<ShortenInstructions>(NeverPrint));

  Manager.registerPass(std::make_unique<RemoveNops>(NeverPrint),
                       !opts::KeepNops);

  Manager.registerPass(std::make_unique<NormalizeCFG>(PrintNormalized));

  Manager.registerPass(std::make_unique<StripRepRet>(NeverPrint),
                       opts::StripRepRet);

  Manager.registerPass(std::make_unique<IdenticalCodeFolding>(PrintICF),
                       opts::ICF);

  Manager.registerPass(
      std::make_unique<SpecializeMemcpy1>(NeverPrint, opts::SpecializeMemcpy1),
      !opts::SpecializeMemcpy1.empty());

  Manager.registerPass(std::make_unique<InlineMemcpy>(NeverPrint),
                       opts::StringOps);

  Manager.registerPass(std::make_unique<IndirectCallPromotion>(PrintICP));

  Manager.registerPass(
      std::make_unique<JTFootprintReduction>(PrintJTFootprintReduction),
      opts::JTFootprintReductionFlag);

  Manager.registerPass(
      std::make_unique<SimplifyRODataLoads>(PrintSimplifyROLoads),
      opts::SimplifyRODataLoads);

  Manager.registerPass(std::make_unique<RegReAssign>(PrintRegReAssign),
                       opts::RegReAssign);

  Manager.registerPass(std::make_unique<Inliner>(PrintInline));

  Manager.registerPass(std::make_unique<IdenticalCodeFolding>(PrintICF),
                       opts::ICF);

  Manager.registerPass(std::make_unique<PLTCall>(PrintPLT));

  Manager.registerPass(std::make_unique<ThreeWayBranch>(),
                       opts::ThreeWayBranchFlag);

  Manager.registerPass(std::make_unique<ReorderBasicBlocks>(PrintReordered));

  Manager.registerPass(std::make_unique<EliminateUnreachableBlocks>(PrintUCE),
                       opts::EliminateUnreachable);

  Manager.registerPass(std::make_unique<SplitFunctions>(PrintSplit));

  Manager.registerPass(std::make_unique<LoopInversionPass>());

  Manager.registerPass(std::make_unique<TailDuplication>());

  Manager.registerPass(std::make_unique<CMOVConversion>(),
                       opts::CMOVConversionFlag);

  // This pass syncs local branches with CFG. If any of the following
  // passes breaks the sync - they either need to re-run the pass or
  // fix branches consistency internally.
  Manager.registerPass(std::make_unique<FixupBranches>(PrintAfterBranchFixup));

  // This pass should come close to last since it uses the estimated hot
  // size of a function to determine the order.  It should definitely
  // also happen after any changes to the call graph are made, e.g. inlining.
  Manager.registerPass(
      std::make_unique<ReorderFunctions>(PrintReorderedFunctions));

  // This is the second run of the SplitFunctions pass required by certain
  // splitting strategies (e.g. cdsplit). Running the SplitFunctions pass again
  // after ReorderFunctions allows the finalized function order to be utilized
  // to make more sophisticated splitting decisions, like hot-warm-cold
  // splitting.
  Manager.registerPass(std::make_unique<SplitFunctions>(PrintSplit));

  // Print final dyno stats right while CFG and instruction analysis are intact.
  Manager.registerPass(
      std::make_unique<DynoStatsPrintPass>(
          InitialDynoStats, "after all optimizations before SCTC and FOP"),
      opts::PrintDynoStats || opts::DynoStatsAll);

  // Add the StokeInfo pass, which extract functions for stoke optimization and
  // get the liveness information for them
  Manager.registerPass(std::make_unique<StokeInfo>(PrintStoke), opts::Stoke);

  // This pass introduces conditional jumps into external functions.
  // Between extending CFG to support this and isolating this pass we chose
  // the latter. Thus this pass will do double jump removal and unreachable
  // code elimination if necessary and won't rely on peepholes/UCE for these
  // optimizations.
  // More generally this pass should be the last optimization pass that
  // modifies branches/control flow.  This pass is run after function
  // reordering so that it can tell whether calls are forward/backward
  // accurately.
  Manager.registerPass(
      std::make_unique<SimplifyConditionalTailCalls>(PrintSCTC),
      opts::SimplifyConditionalTailCalls);

  Manager.registerPass(std::make_unique<Peepholes>(PrintPeepholes));

  Manager.registerPass(std::make_unique<AlignerPass>());

  // Perform reordering on data contained in one or more sections using
  // memory profiling data.
  Manager.registerPass(std::make_unique<ReorderData>());

  if (BC.isAArch64()) {
    Manager.registerPass(std::make_unique<ADRRelaxationPass>());

    // Tighten branches according to offset differences between branch and
    // targets. No extra instructions after this pass, otherwise we may have
    // relocations out of range and crash during linking.
    Manager.registerPass(std::make_unique<LongJmpPass>(PrintLongJmp));
  }

  // This pass should always run last.*
  Manager.registerPass(std::make_unique<FinalizeFunctions>(PrintFinalized));

  // FrameOptimizer has an implicit dependency on FinalizeFunctions.
  // FrameOptimizer move values around and needs to update CFIs. To do this, it
  // must read CFI, interpret it and rewrite it, so CFIs need to be correctly
  // placed according to the final layout.
  Manager.registerPass(std::make_unique<FrameOptimizerPass>(PrintFOP));

  Manager.registerPass(std::make_unique<AllocCombinerPass>(PrintFOP));

  Manager.registerPass(
      std::make_unique<RetpolineInsertion>(PrintRetpolineInsertion));

  // Assign each function an output section.
  Manager.registerPass(std::make_unique<AssignSections>());

  // Patch original function entries
  if (BC.HasRelocations)
    Manager.registerPass(std::make_unique<PatchEntries>());

  // This pass turns tail calls into jumps which makes them invisible to
  // function reordering. It's unsafe to use any CFG or instruction analysis
  // after this point.
  Manager.registerPass(
      std::make_unique<InstructionLowering>(PrintAfterLowering));

  // In non-relocation mode, mark functions that do not fit into their original
  // space as non-simple if we have to (e.g. for correct debug info update).
  // NOTE: this pass depends on finalized code.
  if (!BC.HasRelocations)
    Manager.registerPass(std::make_unique<CheckLargeFunctions>(NeverPrint));

  Manager.registerPass(std::make_unique<LowerAnnotations>(NeverPrint));

  // Check for dirty state of MCSymbols caused by running calculateEmittedSize
  // in parallel and restore them
  Manager.registerPass(std::make_unique<CleanMCState>(NeverPrint));

  return Manager.runPasses();
}

} // namespace bolt
} // namespace llvm
