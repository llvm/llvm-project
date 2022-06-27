//===--- bolt/Passes/Hugify.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/Hugify.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "bolt-hugify"

using namespace llvm;

namespace llvm {
namespace bolt {

void HugePage::runOnFunctions(BinaryContext &BC) {
  auto *RtLibrary = BC.getRuntimeLibrary();
  if (!RtLibrary || !BC.isELF() || !BC.StartFunctionAddress) {
    return;
  }

  auto createSimpleFunction =
      [&](std::string Title, std::vector<MCInst> Instrs) -> BinaryFunction * {
    BinaryFunction *Func = BC.createInjectedBinaryFunction(Title);

    std::vector<std::unique_ptr<BinaryBasicBlock>> BBs;
    BBs.emplace_back(Func->createBasicBlock(nullptr));
    BBs.back()->addInstructions(Instrs.begin(), Instrs.end());
    BBs.back()->setCFIState(0);
    BBs.back()->setOffset(BinaryBasicBlock::INVALID_OFFSET);

    Func->insertBasicBlocks(nullptr, std::move(BBs),
                            /*UpdateLayout=*/true,
                            /*UpdateCFIState=*/false);
    Func->updateState(BinaryFunction::State::CFG_Finalized);
    return Func;
  };

  const BinaryFunction *const Start =
      BC.getBinaryFunctionAtAddress(*BC.StartFunctionAddress);
  assert(Start && "Entry point function not found");
  const MCSymbol *StartSym = Start->getSymbol();
  createSimpleFunction("__bolt_hugify_start_program",
                       BC.MIB->createSymbolTrampoline(StartSym, BC.Ctx.get()));
}
} // namespace bolt
} // namespace llvm