//===- bolt/Passes/AsmDump.cpp - Dump BinaryFunction into assembly --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmDumpPass class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/AsmDump.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include <unordered_set>

#define DEBUG_TYPE "asm-dump"

using namespace llvm;

namespace opts {
extern bool shouldPrint(const bolt::BinaryFunction &Function);
extern cl::OptionCategory BoltCategory;
extern cl::opt<unsigned> Verbosity;

cl::opt<std::string> AsmDump("asm-dump",
                             cl::desc("dump function into assembly"),
                             cl::value_desc("dump folder"), cl::ValueOptional,
                             cl::Hidden, cl::cat(BoltCategory));
} // end namespace opts

namespace llvm {
namespace bolt {

void dumpCFI(const BinaryFunction &BF, const MCInst &Instr, AsmPrinter &MAP) {
  const MCCFIInstruction *CFIInstr = BF.getCFIFor(Instr);
  switch (CFIInstr->getOperation()) {
  // Skip unsupported CFI instructions.
  case MCCFIInstruction::OpRememberState:
  case MCCFIInstruction::OpRestoreState:
    if (opts::Verbosity >= 2)
      errs()
          << "BOLT-WARNING: AsmDump: skipping unsupported CFI instruction in "
          << BF << ".\n";

    return;

  default:
    // Emit regular CFI instructions.
    MAP.emitCFIInstruction(*CFIInstr);
  }
}

void dumpTargetFunctionStub(raw_ostream &OS, const BinaryContext &BC,
                            const MCSymbol *CalleeSymb,
                            const BinarySection *&LastCS) {
  const BinaryFunction *CalleeFunc = BC.getFunctionForSymbol(CalleeSymb);
  if (!CalleeFunc || CalleeFunc->isPLTFunction())
    return;

  if (CalleeFunc->getOriginSection() != LastCS) {
    OS << ".section " << CalleeFunc->getOriginSectionName() << '\n';
    LastCS = CalleeFunc->getOriginSection();
  }
  StringRef CalleeName = CalleeFunc->getOneName();
  OS << ".set \"" << CalleeName << "\", 0\n";
}

void dumpJumpTableSymbols(raw_ostream &OS, const JumpTable *JT, AsmPrinter &MAP,
                          const BinarySection *&LastBS) {
  if (&JT->getSection() != LastBS) {
    OS << ".section " << JT->getSectionName() << '\n';
    LastBS = &JT->getSection();
  }
  OS << "\"" << JT->getName() << "\":\n";
  for (MCSymbol *JTEntry : JT->Entries)
    MAP.OutStreamer->emitSymbolValue(JTEntry, JT->OutputEntrySize);
  OS << '\n';
}

void dumpBinaryDataSymbols(raw_ostream &OS, const BinaryData *BD,
                           const BinarySection *&LastBS) {
  if (BD->isJumpTable())
    return;
  if (&BD->getSection() != LastBS) {
    OS << ".section " << BD->getSectionName() << '\n';
    LastBS = &BD->getSection();
  }
  OS << "\"" << BD->getName() << "\": ";
  OS << '\n';
}

void dumpFunction(const BinaryFunction &BF) {
  const BinaryContext &BC = BF.getBinaryContext();
  if (!opts::shouldPrint(BF))
    return;

  // Make sure the new directory exists, creating it if necessary.
  if (!opts::AsmDump.empty()) {
    if (std::error_code EC = sys::fs::create_directories(opts::AsmDump)) {
      errs() << "BOLT-ERROR: could not create directory '" << opts::AsmDump
             << "': " << EC.message() << '\n';
      exit(1);
    }
  }

  std::string PrintName = BF.getPrintName();
  std::replace(PrintName.begin(), PrintName.end(), '/', '-');
  std::string Filename =
      opts::AsmDump.empty()
          ? (PrintName + ".s")
          : (opts::AsmDump + sys::path::get_separator() + PrintName + ".s")
                .str();
  outs() << "BOLT-INFO: Dumping function assembly to " << Filename << "\n";

  std::error_code EC;
  raw_fd_ostream OS(Filename, EC, sys::fs::OF_None);
  if (EC) {
    errs() << "BOLT-ERROR: " << EC.message() << ", unable to open " << Filename
           << " for output.\n";
    exit(1);
  }
  OS.SetUnbuffered();

  // Create local MC context to isolate the effect of ephemeral assembly
  // emission.
  BinaryContext::IndependentCodeEmitter MCEInstance =
      BC.createIndependentMCCodeEmitter();
  MCContext *LocalCtx = MCEInstance.LocalCtx.get();
  std::unique_ptr<MCAsmBackend> MAB(
      BC.TheTarget->createMCAsmBackend(*BC.STI, *BC.MRI, MCTargetOptions()));
  int AsmPrinterVariant = BC.AsmInfo->getAssemblerDialect();
  MCInstPrinter *InstructionPrinter(BC.TheTarget->createMCInstPrinter(
      *BC.TheTriple, AsmPrinterVariant, *BC.AsmInfo, *BC.MII, *BC.MRI));
  auto FOut = std::make_unique<formatted_raw_ostream>(OS);
  FOut->SetUnbuffered();
  std::unique_ptr<MCStreamer> AsmStreamer(
      createAsmStreamer(*LocalCtx, std::move(FOut),
                        /*isVerboseAsm=*/true,
                        /*useDwarfDirectory=*/false, InstructionPrinter,
                        std::move(MCEInstance.MCE), std::move(MAB),
                        /*ShowInst=*/false));
  AsmStreamer->initSections(true, *BC.STI);
  std::unique_ptr<TargetMachine> TM(BC.TheTarget->createTargetMachine(
      BC.TripleName, "", "", TargetOptions(), std::nullopt));
  std::unique_ptr<AsmPrinter> MAP(
      BC.TheTarget->createAsmPrinter(*TM, std::move(AsmStreamer)));

  StringRef FunctionName = BF.getOneName();
  OS << "  .globl " << FunctionName << '\n';
  OS << "  .type " << FunctionName << ", %function\n";
  OS << FunctionName << ":\n";

  // FDATA for the entry point
  if (uint64_t EntryExecCount = BF.getKnownExecutionCount())
    OS << "# FDATA: 0 [unknown] 0 "
       << "1 " << FunctionName << " 0 "
       << "0 " << EntryExecCount << '\n';

  // Binary data references from the function.
  std::unordered_set<const BinaryData *> BDReferences;
  // Function references from the function (to avoid constructing call graph).
  std::unordered_set<const MCSymbol *> CallReferences;

  MAP->OutStreamer->emitCFIStartProc(/*IsSimple=*/false);
  for (const BinaryBasicBlock *BB : BF.getLayout().blocks()) {
    OS << BB->getName() << ": \n";

    const std::string BranchLabel = Twine(BB->getName(), "_br").str();
    const MCInst *LastInst = BB->getLastNonPseudoInstr();

    for (const MCInst &Instr : *BB) {
      // Dump pseudo instructions (CFI)
      if (BC.MIB->isPseudo(Instr)) {
        if (BC.MIB->isCFI(Instr))
          dumpCFI(BF, Instr, *MAP.get());
        continue;
      }

      // Analyze symbol references (data, functions) from the instruction.
      bool IsCall = BC.MIB->isCall(Instr);
      for (const MCOperand &Operand : MCPlus::primeOperands(Instr)) {
        if (Operand.isExpr() &&
            Operand.getExpr()->getKind() == MCExpr::SymbolRef) {
          std::pair<const MCSymbol *, uint64_t> TSI =
              BC.MIB->getTargetSymbolInfo(Operand.getExpr());
          const MCSymbol *Symbol = TSI.first;
          if (IsCall)
            CallReferences.insert(Symbol);
          else if (const BinaryData *BD =
                       BC.getBinaryDataByName(Symbol->getName()))
            BDReferences.insert(BD);
        }
      }

      if (&Instr == LastInst && (BB->succ_size() || IsCall))
        OS << BranchLabel << ":\n";

      BC.InstPrinter->printInst(&Instr, 0, "", *BC.STI, OS);
      OS << '\n';
    }

    // Dump profile data in FDATA format (as parsed by link_fdata).
    for (const BinaryBasicBlock *Succ : BB->successors()) {
      const BinaryBasicBlock::BinaryBranchInfo BI = BB->getBranchInfo(*Succ);
      if (!BI.MispredictedCount && !BI.Count)
        continue;

      OS << "# FDATA: 1 " << FunctionName << " #" << BranchLabel << "# "
         << "1 " << FunctionName << " #" << Succ->getName() << "# "
         << BI.MispredictedCount << " " << BI.Count << '\n';
    }

    OS << '\n';
  }
  MAP->OutStreamer->emitCFIEndProc();

  OS << ".size " << FunctionName << ", .-" << FunctionName << '\n';

  const BinarySection *LastSection = BF.getOriginSection();
  // Print stubs for all target functions.
  for (const MCSymbol *CalleeSymb : CallReferences)
    dumpTargetFunctionStub(OS, BC, CalleeSymb, LastSection);

  OS << "# Jump tables\n";
  // Print all jump tables.
  for (auto &JTI : BF.jumpTables())
    dumpJumpTableSymbols(OS, JTI.second, *MAP.get(), LastSection);

  OS << "# BinaryData\n";
  // Print data references.
  for (const BinaryData *BD : BDReferences)
    dumpBinaryDataSymbols(OS, BD, LastSection);
}

void AsmDumpPass::runOnFunctions(BinaryContext &BC) {
  for (const auto &BFIt : BC.getBinaryFunctions())
    dumpFunction(BFIt.second);
}

} // namespace bolt
} // namespace llvm
