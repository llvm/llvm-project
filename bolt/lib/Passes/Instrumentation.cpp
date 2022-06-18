//===- bolt/Passes/Instrumentation.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instrumentation class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/Instrumentation.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/RuntimeLibs/InstrumentationRuntimeLibrary.h"
#include "bolt/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include <stack>

#define DEBUG_TYPE "bolt-instrumentation"

using namespace llvm;

namespace opts {
extern cl::OptionCategory BoltInstrCategory;

cl::opt<std::string> InstrumentationFilename(
    "instrumentation-file",
    cl::desc("file name where instrumented profile will be saved (default: "
             "/tmp/prof.fdata)"),
    cl::init("/tmp/prof.fdata"), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<std::string> InstrumentationBinpath(
    "instrumentation-binpath",
    cl::desc("path to instumented binary in case if /proc/self/map_files "
             "is not accessible due to access restriction issues"),
    cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<bool> InstrumentationFileAppendPID(
    "instrumentation-file-append-pid",
    cl::desc("append PID to saved profile file name (default: false)"),
    cl::init(false), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<bool> ConservativeInstrumentation(
    "conservative-instrumentation",
    cl::desc("disable instrumentation optimizations that sacrifice profile "
             "accuracy (for debugging, default: false)"),
    cl::init(false), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<uint32_t> InstrumentationSleepTime(
    "instrumentation-sleep-time",
    cl::desc("interval between profile writes (default: 0 = write only at "
             "program end).  This is useful for service workloads when you "
             "want to dump profile every X minutes or if you are killing the "
             "program and the profile is not being dumped at the end."),
    cl::init(0), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<bool> InstrumentationNoCountersClear(
    "instrumentation-no-counters-clear",
    cl::desc("Don't clear counters across dumps "
             "(use with instrumentation-sleep-time option)"),
    cl::init(false), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<bool> InstrumentationWaitForks(
    "instrumentation-wait-forks",
    cl::desc("Wait until all forks of instrumented process will finish "
             "(use with instrumentation-sleep-time option)"),
    cl::init(false), cl::Optional, cl::cat(BoltInstrCategory));

cl::opt<bool>
    InstrumentHotOnly("instrument-hot-only",
                      cl::desc("only insert instrumentation on hot functions "
                               "(needs profile, default: false)"),
                      cl::init(false), cl::Optional,
                      cl::cat(BoltInstrCategory));

cl::opt<bool> InstrumentCalls("instrument-calls",
                              cl::desc("record profile for inter-function "
                                       "control flow activity (default: true)"),
                              cl::init(true), cl::Optional,
                              cl::cat(BoltInstrCategory));
} // namespace opts

namespace llvm {
namespace bolt {

uint32_t Instrumentation::getFunctionNameIndex(const BinaryFunction &Function) {
  auto Iter = FuncToStringIdx.find(&Function);
  if (Iter != FuncToStringIdx.end())
    return Iter->second;
  size_t Idx = Summary->StringTable.size();
  FuncToStringIdx.emplace(std::make_pair(&Function, Idx));
  Summary->StringTable.append(getEscapedName(Function.getOneName()));
  Summary->StringTable.append(1, '\0');
  return Idx;
}

bool Instrumentation::createCallDescription(FunctionDescription &FuncDesc,
                                            const BinaryFunction &FromFunction,
                                            uint32_t From, uint32_t FromNodeID,
                                            const BinaryFunction &ToFunction,
                                            uint32_t To, bool IsInvoke) {
  CallDescription CD;
  // Ordinarily, we don't augment direct calls with an explicit counter, except
  // when forced to do so or when we know this callee could be throwing
  // exceptions, in which case there is no other way to accurately record its
  // frequency.
  bool ForceInstrumentation = opts::ConservativeInstrumentation || IsInvoke;
  CD.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  CD.FromLoc.Offset = From;
  CD.FromNode = FromNodeID;
  CD.Target = &ToFunction;
  CD.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  CD.ToLoc.Offset = To;
  CD.Counter = ForceInstrumentation ? Summary->Counters.size() : 0xffffffff;
  if (ForceInstrumentation)
    ++DirectCallCounters;
  FuncDesc.Calls.emplace_back(CD);
  return ForceInstrumentation;
}

void Instrumentation::createIndCallDescription(
    const BinaryFunction &FromFunction, uint32_t From) {
  IndCallDescription ICD;
  ICD.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  ICD.FromLoc.Offset = From;
  Summary->IndCallDescriptions.emplace_back(ICD);
}

void Instrumentation::createIndCallTargetDescription(
    const BinaryFunction &ToFunction, uint32_t To) {
  IndCallTargetDescription ICD;
  ICD.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  ICD.ToLoc.Offset = To;
  ICD.Target = &ToFunction;
  Summary->IndCallTargetDescriptions.emplace_back(ICD);
}

bool Instrumentation::createEdgeDescription(FunctionDescription &FuncDesc,
                                            const BinaryFunction &FromFunction,
                                            uint32_t From, uint32_t FromNodeID,
                                            const BinaryFunction &ToFunction,
                                            uint32_t To, uint32_t ToNodeID,
                                            bool Instrumented) {
  EdgeDescription ED;
  auto Result = FuncDesc.EdgesSet.insert(std::make_pair(FromNodeID, ToNodeID));
  // Avoid creating duplicated edge descriptions. This happens in CFGs where a
  // block jumps to its fall-through.
  if (Result.second == false)
    return false;
  ED.FromLoc.FuncString = getFunctionNameIndex(FromFunction);
  ED.FromLoc.Offset = From;
  ED.FromNode = FromNodeID;
  ED.ToLoc.FuncString = getFunctionNameIndex(ToFunction);
  ED.ToLoc.Offset = To;
  ED.ToNode = ToNodeID;
  ED.Counter = Instrumented ? Summary->Counters.size() : 0xffffffff;
  if (Instrumented)
    ++BranchCounters;
  FuncDesc.Edges.emplace_back(ED);
  return Instrumented;
}

void Instrumentation::createLeafNodeDescription(FunctionDescription &FuncDesc,
                                                uint32_t Node) {
  InstrumentedNode IN;
  IN.Node = Node;
  IN.Counter = Summary->Counters.size();
  ++LeafNodeCounters;
  FuncDesc.LeafNodes.emplace_back(IN);
}

InstructionListType
Instrumentation::createInstrumentationSnippet(BinaryContext &BC, bool IsLeaf) {
  auto L = BC.scopeLock();
  MCSymbol *Label;
  Label = BC.Ctx->createNamedTempSymbol("InstrEntry");
  Summary->Counters.emplace_back(Label);
  InstructionListType CounterInstrs;
  BC.MIB->createInstrIncMemory(CounterInstrs, Label, &*BC.Ctx, IsLeaf);
  return CounterInstrs;
}

namespace {

// Helper instruction sequence insertion function
BinaryBasicBlock::iterator insertInstructions(InstructionListType &Instrs,
                                              BinaryBasicBlock &BB,
                                              BinaryBasicBlock::iterator Iter) {
  for (MCInst &NewInst : Instrs) {
    Iter = BB.insertInstruction(Iter, NewInst);
    ++Iter;
  }
  return Iter;
}
} // namespace

void Instrumentation::instrumentLeafNode(BinaryBasicBlock &BB,
                                         BinaryBasicBlock::iterator Iter,
                                         bool IsLeaf,
                                         FunctionDescription &FuncDesc,
                                         uint32_t Node) {
  createLeafNodeDescription(FuncDesc, Node);
  InstructionListType CounterInstrs = createInstrumentationSnippet(
      BB.getFunction()->getBinaryContext(), IsLeaf);
  insertInstructions(CounterInstrs, BB, Iter);
}

void Instrumentation::instrumentIndirectTarget(BinaryBasicBlock &BB,
                                               BinaryBasicBlock::iterator &Iter,
                                               BinaryFunction &FromFunction,
                                               uint32_t From) {
  auto L = FromFunction.getBinaryContext().scopeLock();
  const size_t IndCallSiteID = Summary->IndCallDescriptions.size();
  createIndCallDescription(FromFunction, From);

  BinaryContext &BC = FromFunction.getBinaryContext();
  bool IsTailCall = BC.MIB->isTailCall(*Iter);
  InstructionListType CounterInstrs = BC.MIB->createInstrumentedIndirectCall(
      *Iter, IsTailCall,
      IsTailCall ? IndTailCallHandlerExitBBFunction->getSymbol()
                 : IndCallHandlerExitBBFunction->getSymbol(),
      IndCallSiteID, &*BC.Ctx);

  Iter = BB.eraseInstruction(Iter);
  Iter = insertInstructions(CounterInstrs, BB, Iter);
  --Iter;
}

bool Instrumentation::instrumentOneTarget(
    SplitWorklistTy &SplitWorklist, SplitInstrsTy &SplitInstrs,
    BinaryBasicBlock::iterator &Iter, BinaryFunction &FromFunction,
    BinaryBasicBlock &FromBB, uint32_t From, BinaryFunction &ToFunc,
    BinaryBasicBlock *TargetBB, uint32_t ToOffset, bool IsLeaf, bool IsInvoke,
    FunctionDescription *FuncDesc, uint32_t FromNodeID, uint32_t ToNodeID) {
  {
    auto L = FromFunction.getBinaryContext().scopeLock();
    bool Created = true;
    if (!TargetBB)
      Created = createCallDescription(*FuncDesc, FromFunction, From, FromNodeID,
                                      ToFunc, ToOffset, IsInvoke);
    else
      Created = createEdgeDescription(*FuncDesc, FromFunction, From, FromNodeID,
                                      ToFunc, ToOffset, ToNodeID,
                                      /*Instrumented=*/true);
    if (!Created)
      return false;
  }

  InstructionListType CounterInstrs =
      createInstrumentationSnippet(FromFunction.getBinaryContext(), IsLeaf);

  BinaryContext &BC = FromFunction.getBinaryContext();
  const MCInst &Inst = *Iter;
  if (BC.MIB->isCall(Inst)) {
    // This code handles both
    // - (regular) inter-function calls (cross-function control transfer),
    // - (rare) intra-function calls (function-local control transfer)
    Iter = insertInstructions(CounterInstrs, FromBB, Iter);
    return true;
  }

  if (!TargetBB || !FuncDesc)
    return false;

  // Indirect branch, conditional branches or fall-throughs
  // Regular cond branch, put counter at start of target block
  //
  // N.B.: (FromBB != TargetBBs) checks below handle conditional jumps where
  // we can't put the instrumentation counter in this block because not all
  // paths that reach it at this point will be taken and going to the target.
  if (TargetBB->pred_size() == 1 && &FromBB != TargetBB &&
      !TargetBB->isEntryPoint()) {
    insertInstructions(CounterInstrs, *TargetBB, TargetBB->begin());
    return true;
  }
  if (FromBB.succ_size() == 1 && &FromBB != TargetBB) {
    Iter = insertInstructions(CounterInstrs, FromBB, Iter);
    return true;
  }
  // Critical edge, create BB and put counter there
  SplitWorklist.emplace_back(&FromBB, TargetBB);
  SplitInstrs.emplace_back(std::move(CounterInstrs));
  return true;
}

void Instrumentation::instrumentFunction(BinaryFunction &Function,
                                         MCPlusBuilder::AllocatorIdTy AllocId) {
  if (Function.hasUnknownControlFlow())
    return;

  BinaryContext &BC = Function.getBinaryContext();
  if (BC.isMachO() && Function.hasName("___GLOBAL_init_65535/1"))
    return;

  SplitWorklistTy SplitWorklist;
  SplitInstrsTy SplitInstrs;

  FunctionDescription *FuncDesc = nullptr;
  {
    std::unique_lock<std::shared_timed_mutex> L(FDMutex);
    Summary->FunctionDescriptions.emplace_back();
    FuncDesc = &Summary->FunctionDescriptions.back();
  }

  FuncDesc->Function = &Function;
  Function.disambiguateJumpTables(AllocId);
  Function.deleteConservativeEdges();

  std::unordered_map<const BinaryBasicBlock *, uint32_t> BBToID;
  uint32_t Id = 0;
  for (auto BBI = Function.begin(); BBI != Function.end(); ++BBI) {
    BBToID[&*BBI] = Id++;
  }
  std::unordered_set<const BinaryBasicBlock *> VisitedSet;
  // DFS to establish edges we will use for a spanning tree. Edges in the
  // spanning tree can be instrumentation-free since their count can be
  // inferred by solving flow equations on a bottom-up traversal of the tree.
  // Exit basic blocks are always instrumented so we start the traversal with
  // a minimum number of defined variables to make the equation solvable.
  std::stack<std::pair<const BinaryBasicBlock *, BinaryBasicBlock *>> Stack;
  std::unordered_map<const BinaryBasicBlock *,
                     std::set<const BinaryBasicBlock *>>
      STOutSet;
  for (auto BBI = Function.layout_rbegin(); BBI != Function.layout_rend();
       ++BBI) {
    if ((*BBI)->isEntryPoint() || (*BBI)->isLandingPad()) {
      Stack.push(std::make_pair(nullptr, *BBI));
      if (opts::InstrumentCalls && (*BBI)->isEntryPoint()) {
        EntryNode E;
        E.Node = BBToID[&**BBI];
        E.Address = (*BBI)->getInputOffset();
        FuncDesc->EntryNodes.emplace_back(E);
        createIndCallTargetDescription(Function, (*BBI)->getInputOffset());
      }
    }
  }

  // Modified version of BinaryFunction::dfs() to build a spanning tree
  if (!opts::ConservativeInstrumentation) {
    while (!Stack.empty()) {
      BinaryBasicBlock *BB;
      const BinaryBasicBlock *Pred;
      std::tie(Pred, BB) = Stack.top();
      Stack.pop();
      if (VisitedSet.find(BB) != VisitedSet.end())
        continue;

      VisitedSet.insert(BB);
      if (Pred)
        STOutSet[Pred].insert(BB);

      for (BinaryBasicBlock *SuccBB : BB->successors())
        Stack.push(std::make_pair(BB, SuccBB));
    }
  }

  // Determine whether this is a leaf function, which needs special
  // instructions to protect the red zone
  bool IsLeafFunction = true;
  DenseSet<const BinaryBasicBlock *> InvokeBlocks;
  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    for (auto I = BBI->begin(), E = BBI->end(); I != E; ++I) {
      if (BC.MIB->isCall(*I)) {
        if (BC.MIB->isInvoke(*I))
          InvokeBlocks.insert(&*BBI);
        IsLeafFunction = false;
      }
    }
  }

  for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
    BinaryBasicBlock &BB = *BBI;
    bool HasUnconditionalBranch = false;
    bool HasJumpTable = false;
    bool IsInvokeBlock = InvokeBlocks.count(&BB) > 0;

    for (auto I = BB.begin(); I != BB.end(); ++I) {
      const MCInst &Inst = *I;
      if (!BC.MIB->getOffset(Inst))
        continue;

      const bool IsJumpTable = Function.getJumpTable(Inst);
      if (IsJumpTable)
        HasJumpTable = true;
      else if (BC.MIB->isUnconditionalBranch(Inst))
        HasUnconditionalBranch = true;
      else if ((!BC.MIB->isCall(Inst) && !BC.MIB->isConditionalBranch(Inst)) ||
               BC.MIB->isUnsupportedBranch(Inst.getOpcode()))
        continue;

      const uint32_t FromOffset = *BC.MIB->getOffset(Inst);
      const MCSymbol *Target = BC.MIB->getTargetSymbol(Inst);
      BinaryBasicBlock *TargetBB = Function.getBasicBlockForLabel(Target);
      uint32_t ToOffset = TargetBB ? TargetBB->getInputOffset() : 0;
      BinaryFunction *TargetFunc =
          TargetBB ? &Function : BC.getFunctionForSymbol(Target);
      if (TargetFunc && BC.MIB->isCall(Inst)) {
        if (opts::InstrumentCalls) {
          const BinaryBasicBlock *ForeignBB =
              TargetFunc->getBasicBlockForLabel(Target);
          if (ForeignBB)
            ToOffset = ForeignBB->getInputOffset();
          instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                              FromOffset, *TargetFunc, TargetBB, ToOffset,
                              IsLeafFunction, IsInvokeBlock, FuncDesc,
                              BBToID[&BB]);
        }
        continue;
      }
      if (TargetFunc) {
        // Do not instrument edges in the spanning tree
        if (STOutSet[&BB].find(TargetBB) != STOutSet[&BB].end()) {
          auto L = BC.scopeLock();
          createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                                Function, ToOffset, BBToID[TargetBB],
                                /*Instrumented=*/false);
          continue;
        }
        instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                            FromOffset, *TargetFunc, TargetBB, ToOffset,
                            IsLeafFunction, IsInvokeBlock, FuncDesc,
                            BBToID[&BB], BBToID[TargetBB]);
        continue;
      }

      if (IsJumpTable) {
        for (BinaryBasicBlock *&Succ : BB.successors()) {
          // Do not instrument edges in the spanning tree
          if (STOutSet[&BB].find(&*Succ) != STOutSet[&BB].end()) {
            auto L = BC.scopeLock();
            createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                                  Function, Succ->getInputOffset(),
                                  BBToID[&*Succ], /*Instrumented=*/false);
            continue;
          }
          instrumentOneTarget(
              SplitWorklist, SplitInstrs, I, Function, BB, FromOffset, Function,
              &*Succ, Succ->getInputOffset(), IsLeafFunction, IsInvokeBlock,
              FuncDesc, BBToID[&BB], BBToID[&*Succ]);
        }
        continue;
      }

      // Handle indirect calls -- could be direct calls with unknown targets
      // or secondary entry points of known functions, so check it is indirect
      // to be sure.
      if (opts::InstrumentCalls && BC.MIB->isIndirectCall(*I))
        instrumentIndirectTarget(BB, I, Function, FromOffset);

    } // End of instructions loop

    // Instrument fallthroughs (when the direct jump instruction is missing)
    if (!HasUnconditionalBranch && !HasJumpTable && BB.succ_size() > 0 &&
        BB.size() > 0) {
      BinaryBasicBlock *FTBB = BB.getFallthrough();
      assert(FTBB && "expected valid fall-through basic block");
      auto I = BB.begin();
      auto LastInstr = BB.end();
      --LastInstr;
      while (LastInstr != I && BC.MIB->isPseudo(*LastInstr))
        --LastInstr;
      uint32_t FromOffset = 0;
      // The last instruction in the BB should have an annotation, except
      // if it was branching to the end of the function as a result of
      // __builtin_unreachable(), in which case it was deleted by fixBranches.
      // Ignore this case. FIXME: force fixBranches() to preserve the offset.
      if (!BC.MIB->getOffset(*LastInstr))
        continue;
      FromOffset = *BC.MIB->getOffset(*LastInstr);

      // Do not instrument edges in the spanning tree
      if (STOutSet[&BB].find(FTBB) != STOutSet[&BB].end()) {
        auto L = BC.scopeLock();
        createEdgeDescription(*FuncDesc, Function, FromOffset, BBToID[&BB],
                              Function, FTBB->getInputOffset(), BBToID[FTBB],
                              /*Instrumented=*/false);
        continue;
      }
      instrumentOneTarget(SplitWorklist, SplitInstrs, I, Function, BB,
                          FromOffset, Function, FTBB, FTBB->getInputOffset(),
                          IsLeafFunction, IsInvokeBlock, FuncDesc, BBToID[&BB],
                          BBToID[FTBB]);
    }
  } // End of BBs loop

  // Instrument spanning tree leaves
  if (!opts::ConservativeInstrumentation) {
    for (auto BBI = Function.begin(), BBE = Function.end(); BBI != BBE; ++BBI) {
      BinaryBasicBlock &BB = *BBI;
      if (STOutSet[&BB].size() == 0)
        instrumentLeafNode(BB, BB.begin(), IsLeafFunction, *FuncDesc,
                           BBToID[&BB]);
    }
  }

  // Consume list of critical edges: split them and add instrumentation to the
  // newly created BBs
  auto Iter = SplitInstrs.begin();
  for (std::pair<BinaryBasicBlock *, BinaryBasicBlock *> &BBPair :
       SplitWorklist) {
    BinaryBasicBlock *NewBB = Function.splitEdge(BBPair.first, BBPair.second);
    NewBB->addInstructions(Iter->begin(), Iter->end());
    ++Iter;
  }

  // Unused now
  FuncDesc->EdgesSet.clear();
}

void Instrumentation::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/false,
                                                 /*IsText=*/false,
                                                 /*IsAllocatable=*/true);
  BC.registerOrUpdateSection(".bolt.instr.counters", ELF::SHT_PROGBITS, Flags,
                             nullptr, 0, 1);

  BC.registerOrUpdateNoteSection(".bolt.instr.tables", nullptr, 0,
                                 /*Alignment=*/1,
                                 /*IsReadOnly=*/true, ELF::SHT_NOTE);

  Summary->IndCallCounterFuncPtr =
      BC.Ctx->getOrCreateSymbol("__bolt_ind_call_counter_func_pointer");
  Summary->IndTailCallCounterFuncPtr =
      BC.Ctx->getOrCreateSymbol("__bolt_ind_tailcall_counter_func_pointer");

  createAuxiliaryFunctions(BC);

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return (!BF.isSimple() || BF.isIgnored() ||
            (opts::InstrumentHotOnly && !BF.getKnownExecutionCount()));
  };

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocatorId) {
        instrumentFunction(BF, AllocatorId);
      };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      BC, ParallelUtilities::SchedulingPolicy::SP_INST_QUADRATIC, WorkFun,
      SkipPredicate, "instrumentation", /* ForceSequential=*/true);

  if (BC.isMachO()) {
    if (BC.StartFunctionAddress) {
      BinaryFunction *Main =
          BC.getBinaryFunctionAtAddress(*BC.StartFunctionAddress);
      assert(Main && "Entry point function not found");
      BinaryBasicBlock &BB = Main->front();

      ErrorOr<BinarySection &> SetupSection =
          BC.getUniqueSectionByName("I__setup");
      if (!SetupSection) {
        llvm::errs() << "Cannot find I__setup section\n";
        exit(1);
      }
      MCSymbol *Target = BC.registerNameAtAddress(
          "__bolt_instr_setup", SetupSection->getAddress(), 0, 0);
      MCInst NewInst;
      BC.MIB->createCall(NewInst, Target, BC.Ctx.get());
      BB.insertInstruction(BB.begin(), std::move(NewInst));
    } else {
      llvm::errs() << "BOLT-WARNING: Entry point not found\n";
    }

    if (BinaryData *BD = BC.getBinaryDataByName("___GLOBAL_init_65535/1")) {
      BinaryFunction *Ctor = BC.getBinaryFunctionAtAddress(BD->getAddress());
      assert(Ctor && "___GLOBAL_init_65535 function not found");
      BinaryBasicBlock &BB = Ctor->front();
      ErrorOr<BinarySection &> FiniSection =
          BC.getUniqueSectionByName("I__fini");
      if (!FiniSection) {
        llvm::errs() << "Cannot find I__fini section\n";
        exit(1);
      }
      MCSymbol *Target = BC.registerNameAtAddress(
          "__bolt_instr_fini", FiniSection->getAddress(), 0, 0);
      auto IsLEA = [&BC](const MCInst &Inst) { return BC.MIB->isLEA64r(Inst); };
      const auto LEA =
          std::find_if(std::next(std::find_if(BB.rbegin(), BB.rend(), IsLEA)),
                       BB.rend(), IsLEA);
      LEA->getOperand(4).setExpr(
          MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *BC.Ctx));
    } else {
      llvm::errs() << "BOLT-WARNING: ___GLOBAL_init_65535 not found\n";
    }
  }

  setupRuntimeLibrary(BC);
}

void Instrumentation::createAuxiliaryFunctions(BinaryContext &BC) {
  auto createSimpleFunction =
      [&](StringRef Title, InstructionListType Instrs) -> BinaryFunction * {
    BinaryFunction *Func = BC.createInjectedBinaryFunction(std::string(Title));

    std::vector<std::unique_ptr<BinaryBasicBlock>> BBs;
    BBs.emplace_back(Func->createBasicBlock());
    BBs.back()->addInstructions(Instrs.begin(), Instrs.end());
    BBs.back()->setCFIState(0);
    Func->insertBasicBlocks(nullptr, std::move(BBs),
                            /*UpdateLayout=*/true,
                            /*UpdateCFIState=*/false);
    Func->updateState(BinaryFunction::State::CFG_Finalized);
    return Func;
  };

  // Here we are creating a set of functions to handle BB entry/exit.
  // IndCallHandlerExitBB contains instructions to finish handling traffic to an
  // indirect call. We pass it to createInstrumentedIndCallHandlerEntryBB(),
  // which will check if a pointer to runtime library traffic accounting
  // function was initialized (it is done during initialization of runtime
  // library). If it is so - calls it. Then this routine returns to normal
  // execution by jumping to exit BB.
  BinaryFunction *IndCallHandlerExitBB =
      createSimpleFunction("__bolt_instr_ind_call_handler",
                           BC.MIB->createInstrumentedIndCallHandlerExitBB());

  IndCallHandlerExitBBFunction =
      createSimpleFunction("__bolt_instr_ind_call_handler_func",
                           BC.MIB->createInstrumentedIndCallHandlerEntryBB(
                               Summary->IndCallCounterFuncPtr,
                               IndCallHandlerExitBB->getSymbol(), &*BC.Ctx));

  BinaryFunction *IndTailCallHandlerExitBB = createSimpleFunction(
      "__bolt_instr_ind_tail_call_handler",
      BC.MIB->createInstrumentedIndTailCallHandlerExitBB());

  IndTailCallHandlerExitBBFunction = createSimpleFunction(
      "__bolt_instr_ind_tailcall_handler_func",
      BC.MIB->createInstrumentedIndCallHandlerEntryBB(
          Summary->IndTailCallCounterFuncPtr,
          IndTailCallHandlerExitBB->getSymbol(), &*BC.Ctx));

  createSimpleFunction("__bolt_num_counters_getter",
                       BC.MIB->createNumCountersGetter(BC.Ctx.get()));
  createSimpleFunction("__bolt_instr_locations_getter",
                       BC.MIB->createInstrLocationsGetter(BC.Ctx.get()));
  createSimpleFunction("__bolt_instr_tables_getter",
                       BC.MIB->createInstrTablesGetter(BC.Ctx.get()));
  createSimpleFunction("__bolt_instr_num_funcs_getter",
                       BC.MIB->createInstrNumFuncsGetter(BC.Ctx.get()));

  if (BC.isELF()) {
    if (BC.StartFunctionAddress) {
      BinaryFunction *Start =
          BC.getBinaryFunctionAtAddress(*BC.StartFunctionAddress);
      assert(Start && "Entry point function not found");
      const MCSymbol *StartSym = Start->getSymbol();
      createSimpleFunction(
          "__bolt_start_trampoline",
          BC.MIB->createSymbolTrampoline(StartSym, BC.Ctx.get()));
    }
    if (BC.FiniFunctionAddress) {
      BinaryFunction *Fini =
          BC.getBinaryFunctionAtAddress(*BC.FiniFunctionAddress);
      assert(Fini && "Finalization function not found");
      const MCSymbol *FiniSym = Fini->getSymbol();
      createSimpleFunction(
          "__bolt_fini_trampoline",
          BC.MIB->createSymbolTrampoline(FiniSym, BC.Ctx.get()));
    } else {
      // Create dummy return function for trampoline to avoid issues
      // with unknown symbol in runtime library. E.g. for static PIE
      // executable
      createSimpleFunction("__bolt_fini_trampoline",
                           BC.MIB->createDummyReturnFunction(BC.Ctx.get()));
    }
  }
}

void Instrumentation::setupRuntimeLibrary(BinaryContext &BC) {
  uint32_t FuncDescSize = Summary->getFDSize();

  outs() << "BOLT-INSTRUMENTER: Number of indirect call site descriptors: "
         << Summary->IndCallDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of indirect call target descriptors: "
         << Summary->IndCallTargetDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of function descriptors: "
         << Summary->FunctionDescriptions.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of branch counters: " << BranchCounters
         << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of ST leaf node counters: "
         << LeafNodeCounters << "\n";
  outs() << "BOLT-INSTRUMENTER: Number of direct call counters: "
         << DirectCallCounters << "\n";
  outs() << "BOLT-INSTRUMENTER: Total number of counters: "
         << Summary->Counters.size() << "\n";
  outs() << "BOLT-INSTRUMENTER: Total size of counters: "
         << (Summary->Counters.size() * 8) << " bytes (static alloc memory)\n";
  outs() << "BOLT-INSTRUMENTER: Total size of string table emitted: "
         << Summary->StringTable.size() << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Total size of descriptors: "
         << (FuncDescSize +
             Summary->IndCallDescriptions.size() * sizeof(IndCallDescription) +
             Summary->IndCallTargetDescriptions.size() *
                 sizeof(IndCallTargetDescription))
         << " bytes in file\n";
  outs() << "BOLT-INSTRUMENTER: Profile will be saved to file "
         << opts::InstrumentationFilename << "\n";

  InstrumentationRuntimeLibrary *RtLibrary =
      static_cast<InstrumentationRuntimeLibrary *>(BC.getRuntimeLibrary());
  assert(RtLibrary && "instrumentation runtime library object must be set");
  RtLibrary->setSummary(std::move(Summary));
}
} // namespace bolt
} // namespace llvm
