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
#include "Driver.h"
#include "Error.h"
#include "InputFiles.h"
#include "Symbols.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/ParallelCG.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/IRMover.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// This is for use when debugging LTO.
static void saveLtoObjectFile(StringRef Buffer, unsigned I, bool Many) {
  SmallString<128> Filename = Config->OutputFile;
  if (Many)
    Filename += utostr(I);
  Filename += ".lto.o";
  std::error_code EC;
  raw_fd_ostream OS(Filename, EC, sys::fs::OpenFlags::F_None);
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

static bool shouldInternalize(const SmallPtrSet<GlobalValue *, 8> &Used,
                              Symbol *S, GlobalValue *GV) {
  if (S->IsUsedInRegularObj)
    return false;

  if (Used.count(GV))
    return false;

  return !S->includeInDynsym();
}

BitcodeCompiler::BitcodeCompiler()
    : Combined(new llvm::Module("ld-temp.o", Driver->Context)),
      Mover(*Combined) {}

void BitcodeCompiler::add(BitcodeFile &F) {
  std::unique_ptr<IRObjectFile> Obj = std::move(F.Obj);
  std::vector<GlobalValue *> Keep;
  unsigned BodyIndex = 0;
  ArrayRef<Symbol *> Syms = F.getSymbols();

  Module &M = Obj->getModule();
  if (M.getDataLayoutStr().empty())
    fatal("invalid bitcode file: " + F.getName() + " has no datalayout");

  // If a symbol appears in @llvm.used, the linker is required
  // to treat the symbol as there is a reference to the symbol
  // that it cannot see. Therefore, we can't internalize.
  SmallPtrSet<GlobalValue *, 8> Used;
  collectUsedGlobalVariables(M, Used, /* CompilerUsed */ false);

  // This function is called if we know that the combined LTO object will
  // provide a definition of a symbol. It undefines the symbol so that the
  // definition in the combined LTO object will replace it when parsed.
  auto Undefine = [](Symbol *S) {
    replaceBody<Undefined>(S, S->body()->getName(), STV_DEFAULT, 0);
  };

  for (const BasicSymbolRef &Sym : Obj->symbols()) {
    uint32_t Flags = Sym.getFlags();
    GlobalValue *GV = Obj->getSymbolGV(Sym.getRawDataRefImpl());
    if (GV && GV->hasAppendingLinkage())
      Keep.push_back(GV);
    if (BitcodeFile::shouldSkip(Flags))
      continue;
    Symbol *S = Syms[BodyIndex++];
    if (Flags & BasicSymbolRef::SF_Undefined)
      continue;
    if (!GV) {
      // Module asm symbol.
      Undefine(S);
      continue;
    }
    auto *B = dyn_cast<DefinedBitcode>(S->body());
    if (!B || B->File != &F)
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
    if (shouldInternalize(Used, S, GV))
      InternalizedSyms.insert(GV->getName());
    else
      Undefine(S);

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

std::vector<std::unique_ptr<InputFile>> BitcodeCompiler::runSplitCodegen(
    const std::function<std::unique_ptr<TargetMachine>()> &TMFactory) {
  unsigned NumThreads = Config->LtoJobs;
  OwningData.resize(NumThreads);

  std::list<raw_svector_ostream> OSs;
  std::vector<raw_pwrite_stream *> OSPtrs;
  for (SmallString<0> &Obj : OwningData) {
    OSs.emplace_back(Obj);
    OSPtrs.push_back(&OSs.back());
  }

  splitCodeGen(std::move(Combined), OSPtrs, {}, TMFactory);

  std::vector<std::unique_ptr<InputFile>> ObjFiles;
  for (SmallString<0> &Obj : OwningData)
    ObjFiles.push_back(createObjectFile(
        MemoryBufferRef(Obj, "LLD-INTERNAL-combined-lto-object")));

  if (Config->SaveTemps)
    for (unsigned I = 0; I < NumThreads; ++I)
      saveLtoObjectFile(OwningData[I], I, NumThreads > 1);

  return ObjFiles;
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile.
std::vector<std::unique_ptr<InputFile>> BitcodeCompiler::compile() {
  TheTriple = Combined->getTargetTriple();
  for (const auto &Name : InternalizedSyms) {
    GlobalValue *GV = Combined->getNamedValue(Name.first());
    assert(GV);
    internalize(*GV);
  }

  if (Config->SaveTemps)
    saveBCFile(*Combined, ".lto.bc");

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(TheTriple, Msg);
  if (!T)
    fatal("target not found: " + Msg);
  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  Reloc::Model R = Config->Pic ? Reloc::PIC_ : Reloc::Static;

  auto CreateTargetMachine = [&]() {
    return std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TheTriple, "", "", Options, R));
  };

  std::unique_ptr<TargetMachine> TM = CreateTargetMachine();
  runLTOPasses(*Combined, *TM);

  return runSplitCodegen(CreateTargetMachine);
}
