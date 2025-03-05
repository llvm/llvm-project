//===- bolt/Passes/PLTCall.h - PLT call optimization ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PLTCall class, which replaces calls to PLT entries
// with indirect calls against GOT.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/PLTCall.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "bolt-plt"

using namespace llvm;

namespace opts {

extern cl::OptionCategory BoltOptCategory;

cl::opt<bolt::PLTCall::OptType>
PLT("plt",
  cl::desc("optimize PLT calls (requires linking with -znow)"),
  cl::init(bolt::PLTCall::OT_NONE),
  cl::values(clEnumValN(bolt::PLTCall::OT_NONE,
      "none",
      "do not optimize PLT calls"),
    clEnumValN(bolt::PLTCall::OT_HOT,
      "hot",
      "optimize executed (hot) PLT calls"),
    clEnumValN(bolt::PLTCall::OT_ALL,
      "all",
      "optimize all PLT calls")),
  cl::ZeroOrMore,
  cl::cat(BoltOptCategory));

}

namespace llvm {
namespace bolt {

Error PLTCall::runOnFunctions(BinaryContext &BC) {
  if (opts::PLT == OT_NONE)
    return Error::success();

  uint64_t NumCallsOptimized = 0;
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!shouldOptimize(Function))
      continue;

    if (opts::PLT == OT_HOT &&
        Function.getExecutionCount() == BinaryFunction::COUNT_NO_PROFILE)
      continue;

    for (BinaryBasicBlock &BB : Function) {
      if (opts::PLT == OT_HOT && !BB.getKnownExecutionCount())
        continue;

      for (auto II = BB.begin(); II != BB.end(); II++) {
        if (!BC.MIB->isCall(*II))
          continue;
        const MCSymbol *CallSymbol = BC.MIB->getTargetSymbol(*II);
        if (!CallSymbol)
          continue;
        const BinaryFunction *CalleeBF = BC.getFunctionForSymbol(CallSymbol);
        if (!CalleeBF || !CalleeBF->isPLTFunction())
          continue;
        const InstructionListType NewCode = BC.MIB->createIndirectPLTCall(
            std::move(*II), CalleeBF->getPLTSymbol(), BC.Ctx.get());
        II = BB.replaceInstruction(II, NewCode);
        assert(!NewCode.empty() && "PLT Call replacement must be non-empty");
        std::advance(II, NewCode.size() - 1);
        BC.MIB->addAnnotation(*II, "PLTCall", true);
        ++NumCallsOptimized;
      }
    }
  }

  if (NumCallsOptimized) {
    BC.RequiresZNow = true;
    BC.outs() << "BOLT-INFO: " << NumCallsOptimized
              << " PLT calls in the binary were optimized.\n";
  }
  return Error::success();
}

} // namespace bolt
} // namespace llvm
