//===- LTO.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LTO.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// This is for use when debugging LTO.
static void saveLtoObjectFile(StringRef Buffer) {
  std::error_code EC;
  raw_fd_ostream OS(Config->OutputFile.str() + ".lto.o", EC,
                    sys::fs::OpenFlags::F_None);
  check(EC);
  OS << Buffer;
}

// This is for use when debugging LTO.
static void saveBCFile(Module &M, StringRef Suffix) {
  std::error_code EC;
  raw_fd_ostream OS(Config->OutputFile.str() + Suffix.str(), EC,
                    sys::fs::OpenFlags::F_None);
  check(EC);
  WriteBitcodeToFile(&M, OS, /* ShouldPreserveUseListOrder */ true);
}

// Run LTO passes.
// Note that the gold plugin has a similar piece of code, so
// it is probably better to move this code to a common place.
static void runLTOPasses(Module &M, TargetMachine &TM) {
  legacy::PassManager LtoPasses;
  LtoPasses.add(createTargetTransformInfoWrapperPass(TM.getTargetIRAnalysis()));
  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM.getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  PMB.VerifyInput = PMB.VerifyOutput = !Config->DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = Config->LtoO;
  PMB.populateLTOPassManager(LtoPasses);
  LtoPasses.run(M);

  if (Config->SaveTemps)
    saveBCFile(M, ".lto.opt.bc");
}

void BitcodeCompiler::add(BitcodeFile &F) {
  std::unique_ptr<IRObjectFile> Obj =
      check(IRObjectFile::create(F.MB, Context));
  std::vector<GlobalValue *> Keep;
  unsigned BodyIndex = 0;
  ArrayRef<SymbolBody *> Bodies = F.getSymbols();

  Module &M = Obj->getModule();

  // If a symbol appears in @llvm.used, the linker is required
  // to treat the symbol as there is a reference to the symbol
  // that it cannot see. Therefore, we can't internalize.
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /* CompilerUsed */ false);

  for (const BasicSymbolRef &Sym : Obj->symbols()) {
    GlobalValue *GV = Obj->getSymbolGV(Sym.getRawDataRefImpl());
    // Ignore module asm symbols.
    if (!GV)
      continue;
    if (GV->hasAppendingLinkage()) {
      Keep.push_back(GV);
      continue;
    }
    if (BitcodeFile::shouldSkip(Sym))
      continue;
    SymbolBody *B = Bodies[BodyIndex++];
    if (!B || &B->repl() != B || !isa<DefinedBitcode>(B))
      continue;
    switch (GV->getLinkage()) {
    default:
      break;
    case llvm::GlobalValue::LinkOnceAnyLinkage:
      GV->setLinkage(GlobalValue::WeakAnyLinkage);
      break;
    case llvm::GlobalValue::LinkOnceODRLinkage:
      GV->setLinkage(GlobalValue::WeakODRLinkage);
      break;
    }

    // We collect the set of symbols we want to internalize here
    // and change the linkage after the IRMover executed, i.e. after
    // we imported the symbols and satisfied undefined references
    // to it. We can't just change linkage here because otherwise
    // the IRMover will just rename the symbol.
    // Shared libraries need to be handled slightly differently.
    // For now, let's be conservative and just never internalize
    // symbols when creating a shared library.
    if (!Config->Shared && !Config->ExportDynamic && !B->isUsedInRegularObj() &&
        !B->MustBeInDynSym)
      if (!Used.count(GV))
        InternalizedSyms.insert(GV->getName());

    Keep.push_back(GV);
  }

  Mover.move(Obj->takeModule(), Keep,
             [](GlobalValue &, IRMover::ValueAdder) {});
}

static void internalize(GlobalValue &GV) {
  assert(!GV.hasLocalLinkage() &&
         "Trying to internalize a symbol with local linkage!");
  GV.setLinkage(GlobalValue::InternalLinkage);
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile.
std::unique_ptr<InputFile> BitcodeCompiler::compile() {
  for (const auto &Name : InternalizedSyms) {
    GlobalValue *GV = Combined.getNamedValue(Name.first());
    assert(GV);
    internalize(*GV);
  }

  if (Config->SaveTemps)
    saveBCFile(Combined, ".lto.bc");

  std::unique_ptr<TargetMachine> TM(getTargetMachine());
  runLTOPasses(Combined, *TM);

  raw_svector_ostream OS(OwningData);
  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, OS,
                              TargetMachine::CGFT_ObjectFile))
    fatal("failed to setup codegen");
  CodeGenPasses.run(Combined);
  MB = MemoryBuffer::getMemBuffer(OwningData,
                                  "LLD-INTERNAL-combined-lto-object", false);
  if (Config->SaveTemps)
    saveLtoObjectFile(MB->getBuffer());
  return createObjectFile(*MB);
}

TargetMachine *BitcodeCompiler::getTargetMachine() {
  StringRef TripleStr = Combined.getTargetTriple();
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(TripleStr, Msg);
  if (!T)
    fatal("target not found: " + Msg);
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  Reloc::Model R = Config->Pic ? Reloc::PIC_ : Reloc::Static;
  return T->createTargetMachine(TripleStr, "", "", Options, R);
}
