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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

// This is for use when debugging LTO.
static void saveBuffer(StringRef Buffer, const Twine &Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path.str(), EC, sys::fs::OpenFlags::F_None);
  if (EC)
    error(EC, "cannot create " + Path);
  OS << Buffer;
}

static void diagnosticHandler(const DiagnosticInfo &DI) {
  SmallString<128> ErrStorage;
  raw_svector_ostream OS(ErrStorage);
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);
  warn(ErrStorage);
}

static void checkError(Error E) {
  handleAllErrors(std::move(E), [&](ErrorInfoBase &EIB) -> Error {
    error(EIB.message());
    return Error::success();
  });
}

static std::unique_ptr<lto::LTO> createLTO() {
  lto::Config Conf;

  // LLD supports the new relocations.
  Conf.Options = InitTargetOptionsFromCodeGenFlags();
  Conf.Options.RelaxELFRelocations = true;

  Conf.RelocModel = Config->Pic ? Reloc::PIC_ : Reloc::Static;
  Conf.DisableVerify = Config->DisableVerify;
  Conf.DiagHandler = diagnosticHandler;
  Conf.OptLevel = Config->LtoO;

  // Set up a custom pipeline if we've been asked to.
  Conf.OptPipeline = Config->LtoNewPmPasses;
  Conf.AAPipeline = Config->LtoAAPipeline;

  if (Config->SaveTemps)
    checkError(Conf.addSaveTemps(std::string(Config->OutputFile) + ".",
                                 /*UseInputModulePath*/ true));

  lto::ThinBackend Backend;
  if (Config->ThinLtoJobs != -1u)
    Backend = lto::createInProcessThinBackend(Config->ThinLtoJobs);
  return llvm::make_unique<lto::LTO>(std::move(Conf), Backend,
                                     Config->LtoPartitions);
}

BitcodeCompiler::BitcodeCompiler() : LtoObj(createLTO()) {}

BitcodeCompiler::~BitcodeCompiler() = default;

static void undefine(Symbol *S) {
  replaceBody<Undefined>(S, S->body()->getName(), STV_DEFAULT, S->body()->Type,
                         nullptr);
}

void BitcodeCompiler::add(BitcodeFile &F) {
  lto::InputFile &Obj = *F.Obj;
  if (Obj.getDataLayoutStr().empty())
    fatal("invalid bitcode file: " + F.getName() + " has no datalayout");

  unsigned SymNum = 0;
  std::vector<Symbol *> Syms = F.getSymbols();
  std::vector<lto::SymbolResolution> Resols(Syms.size());

  // Provide a resolution to the LTO API for each symbol.
  for (const lto::InputFile::Symbol &ObjSym : Obj.symbols()) {
    Symbol *Sym = Syms[SymNum];
    lto::SymbolResolution &R = Resols[SymNum];
    ++SymNum;
    SymbolBody *B = Sym->body();

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    R.Prevailing =
        !(ObjSym.getFlags() & object::BasicSymbolRef::SF_Undefined) &&
        B->File == &F;

    R.VisibleToRegularObj =
        Sym->IsUsedInRegularObj || (R.Prevailing && Sym->includeInDynsym());
    if (R.Prevailing)
      undefine(Sym);
  }
  checkError(LtoObj->add(std::move(F.Obj), Resols));
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile(s).
std::vector<InputFile *> BitcodeCompiler::compile() {
  std::vector<InputFile *> Ret;
  unsigned MaxTasks = LtoObj->getMaxTasks();
  Buff.resize(MaxTasks);

  checkError(LtoObj->run([&](size_t Task) {
    return llvm::make_unique<lto::NativeObjectStream>(
        llvm::make_unique<raw_svector_ostream>(Buff[Task]));
  }));

  for (unsigned I = 0; I != MaxTasks; ++I) {
    if (Buff[I].empty())
      continue;
    if (Config->SaveTemps) {
      if (MaxTasks == 1)
        saveBuffer(Buff[I], Config->OutputFile + ".lto.o");
      else
        saveBuffer(Buff[I], Config->OutputFile + Twine(I) + ".lto.o");
    }
    InputFile *Obj = createObjectFile(MemoryBufferRef(Buff[I], "lto.tmp"));
    Ret.push_back(Obj);
  }
  return Ret;
}
