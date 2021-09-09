//===- LTO.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LTO.h"
#include "Config.h"
#include "InputFiles.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/Args.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/Caching.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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

// Creates an empty file to store a list of object files for final
// linking of distributed ThinLTO.
static std::unique_ptr<raw_fd_ostream> openFile(StringRef file) {
  std::error_code ec;
  auto ret =
      std::make_unique<raw_fd_ostream>(file, ec, sys::fs::OpenFlags::OF_None);
  if (ec) {
    error("cannot open " + file + ": " + ec.message());
    return nullptr;
  }
  return ret;
}

// The merged bitcode after LTO is large. Try opening a file stream that
// supports reading, seeking and writing. Such a file allows BitcodeWriter to
// flush buffered data to reduce memory consumption. If this fails, open a file
// stream that supports only write.
static std::unique_ptr<raw_fd_ostream> openLTOOutputFile(StringRef file) {
  std::error_code ec;
  std::unique_ptr<raw_fd_ostream> fs =
      std::make_unique<raw_fd_stream>(file, ec);
  if (!ec)
    return fs;
  return openFile(file);
}

static std::string getThinLTOOutputFile(StringRef modulePath) {
  return lto::getThinLTOOutputFile(
      std::string(modulePath), std::string(config->thinLTOPrefixReplace.first),
      std::string(config->thinLTOPrefixReplace.second));
}

static lto::Config createConfig() {
  lto::Config c;

  // LLD supports the new relocations and address-significance tables.
  c.Options = initTargetOptionsFromCodeGenFlags();
  c.Options.RelaxELFRelocations = true;
  c.Options.EmitAddrsig = true;

  // Always emit a section per function/datum with LTO.
  c.Options.FunctionSections = true;
  c.Options.DataSections = true;

  // Check if basic block sections must be used.
  // Allowed values for --lto-basic-block-sections are "all", "labels",
  // "<file name specifying basic block ids>", or none.  This is the equivalent
  // of -fbasic-block-sections= flag in clang.
  if (!config->ltoBasicBlockSections.empty()) {
    if (config->ltoBasicBlockSections == "all") {
      c.Options.BBSections = BasicBlockSection::All;
    } else if (config->ltoBasicBlockSections == "labels") {
      c.Options.BBSections = BasicBlockSection::Labels;
    } else if (config->ltoBasicBlockSections == "none") {
      c.Options.BBSections = BasicBlockSection::None;
    } else {
      ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
          MemoryBuffer::getFile(config->ltoBasicBlockSections.str());
      if (!MBOrErr) {
        error("cannot open " + config->ltoBasicBlockSections + ":" +
              MBOrErr.getError().message());
      } else {
        c.Options.BBSectionsFuncListBuf = std::move(*MBOrErr);
      }
      c.Options.BBSections = BasicBlockSection::List;
    }
  }

  c.Options.PseudoProbeForProfiling = config->ltoPseudoProbeForProfiling;
  c.Options.UniqueBasicBlockSectionNames =
      config->ltoUniqueBasicBlockSectionNames;

  if (auto relocModel = getRelocModelFromCMModel())
    c.RelocModel = *relocModel;
  else if (config->relocatable)
    c.RelocModel = None;
  else if (config->isPic)
    c.RelocModel = Reloc::PIC_;
  else
    c.RelocModel = Reloc::Static;

  c.CodeModel = getCodeModelFromCMModel();
  c.DisableVerify = config->disableVerify;
  c.DiagHandler = diagnosticHandler;
  c.OptLevel = config->ltoo;
  c.CPU = getCPUStr();
  c.MAttrs = getMAttrs();
  c.CGOptLevel = args::getCGOptLevel(config->ltoo);

  c.PTO.LoopVectorization = c.OptLevel > 1;
  c.PTO.SLPVectorization = c.OptLevel > 1;

  // Set up a custom pipeline if we've been asked to.
  c.OptPipeline = std::string(config->ltoNewPmPasses);
  c.AAPipeline = std::string(config->ltoAAPipeline);

  // Set up optimization remarks if we've been asked to.
  c.RemarksFilename = std::string(config->optRemarksFilename);
  c.RemarksPasses = std::string(config->optRemarksPasses);
  c.RemarksWithHotness = config->optRemarksWithHotness;
  c.RemarksHotnessThreshold = config->optRemarksHotnessThreshold;
  c.RemarksFormat = std::string(config->optRemarksFormat);

  c.SampleProfile = std::string(config->ltoSampleProfile);
  c.UseNewPM = config->ltoNewPassManager;
  c.DebugPassManager = config->ltoDebugPassManager;
  c.DwoDir = std::string(config->dwoDir);

  c.HasWholeProgramVisibility = config->ltoWholeProgramVisibility;
  c.AlwaysEmitRegularLTOObj = !config->ltoObjPath.empty();

  for (const llvm::StringRef &name : config->thinLTOModulesToCompile)
    c.ThinLTOModulesToCompile.emplace_back(name);

  c.TimeTraceEnabled = config->timeTraceEnabled;
  c.TimeTraceGranularity = config->timeTraceGranularity;

  c.CSIRProfile = std::string(config->ltoCSProfileFile);
  c.RunCSIRInstr = config->ltoCSProfileGenerate;
  c.PGOWarnMismatch = config->ltoPGOWarnMismatch;

  if (config->emitLLVM) {
    c.PostInternalizeModuleHook = [](size_t task, const Module &m) {
      if (std::unique_ptr<raw_fd_ostream> os =
              openLTOOutputFile(config->outputFile))
        WriteBitcodeToFile(m, *os, false);
      return false;
    };
  }

  if (config->ltoEmitAsm)
    c.CGFileType = CGFT_AssemblyFile;

  if (config->saveTemps)
    checkError(c.addSaveTemps(config->outputFile.str() + ".",
                              /*UseInputModulePath*/ true));
  return c;
}

BitcodeCompiler::BitcodeCompiler() {
  // Initialize indexFile.
  if (!config->thinLTOIndexOnlyArg.empty())
    indexFile = openFile(config->thinLTOIndexOnlyArg);

  // Initialize ltoObj.
  lto::ThinBackend backend;
  if (config->thinLTOIndexOnly) {
    auto onIndexWrite = [&](StringRef s) { thinIndices.erase(s); };
    backend = lto::createWriteIndexesThinBackend(
        std::string(config->thinLTOPrefixReplace.first),
        std::string(config->thinLTOPrefixReplace.second),
        config->thinLTOEmitImportsFiles, indexFile.get(), onIndexWrite);
  } else {
    backend = lto::createInProcessThinBackend(
        llvm::heavyweight_hardware_concurrency(config->thinLTOJobs));
  }

  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend,
                                       config->ltoPartitions);

  // Initialize usedStartStop.
  for (Symbol *sym : symtab->symbols()) {
    StringRef s = sym->getName();
    for (StringRef prefix : {"__start_", "__stop_"})
      if (s.startswith(prefix))
        usedStartStop.insert(s.substr(prefix.size()));
  }
}

BitcodeCompiler::~BitcodeCompiler() = default;

void BitcodeCompiler::add(BitcodeFile &f) {
  lto::InputFile &obj = *f.obj;
  bool isExec = !config->shared && !config->relocatable;

  if (config->thinLTOIndexOnly)
    thinIndices.insert(obj.getName());

  ArrayRef<Symbol *> syms = f.getSymbols();
  ArrayRef<lto::InputFile::Symbol> objSyms = obj.symbols();
  std::vector<lto::SymbolResolution> resols(syms.size());

  // Provide a resolution to the LTO API for each symbol.
  for (size_t i = 0, e = syms.size(); i != e; ++i) {
    Symbol *sym = syms[i];
    const lto::InputFile::Symbol &objSym = objSyms[i];
    lto::SymbolResolution &r = resols[i];

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    r.Prevailing = !objSym.isUndefined() && sym->file == &f;

    // We ask LTO to preserve following global symbols:
    // 1) All symbols when doing relocatable link, so that them can be used
    //    for doing final link.
    // 2) Symbols that are used in regular objects.
    // 3) C named sections if we have corresponding __start_/__stop_ symbol.
    // 4) Symbols that are defined in bitcode files and used for dynamic linking.
    r.VisibleToRegularObj = config->relocatable || sym->isUsedInRegularObj ||
                            (r.Prevailing && sym->includeInDynsym()) ||
                            usedStartStop.count(objSym.getSectionName());
    // Identify symbols exported dynamically, and that therefore could be
    // referenced by a shared library not visible to the linker.
    r.ExportDynamic = sym->computeBinding() != STB_LOCAL &&
                      (sym->isExportDynamic(sym->kind(), sym->visibility) ||
                       sym->exportDynamic || sym->inDynamicList);
    const auto *dr = dyn_cast<Defined>(sym);
    r.FinalDefinitionInLinkageUnit =
        (isExec || sym->visibility != STV_DEFAULT) && dr &&
        // Skip absolute symbols from ELF objects, otherwise PC-rel relocations
        // will be generated by for them, triggering linker errors.
        // Symbol section is always null for bitcode symbols, hence the check
        // for isElf(). Skip linker script defined symbols as well: they have
        // no File defined.
        !(dr->section == nullptr && (!sym->file || sym->file->isElf()));

    if (r.Prevailing)
      sym->replace(Undefined{nullptr, sym->getName(), STB_GLOBAL, STV_DEFAULT,
                             sym->type});

    // We tell LTO to not apply interprocedural optimization for wrapped
    // (with --wrap) symbols because otherwise LTO would inline them while
    // their values are still not final.
    r.LinkerRedefined = !sym->canInline;
  }
  checkError(ltoObj->add(std::move(f.obj), resols));
}

// If LazyObjFile has not been added to link, emit empty index files.
// This is needed because this is what GNU gold plugin does and we have a
// distributed build system that depends on that behavior.
static void thinLTOCreateEmptyIndexFiles() {
  for (LazyObjFile *f : lazyObjFiles) {
    if (f->fetched || !isBitcode(f->mb))
      continue;
    std::string path = replaceThinLTOSuffix(getThinLTOOutputFile(f->getName()));
    std::unique_ptr<raw_fd_ostream> os = openFile(path + ".thinlto.bc");
    if (!os)
      continue;

    ModuleSummaryIndex m(/*HaveGVs*/ false);
    m.setSkipModuleByDistributedBackend();
    WriteIndexToFile(m, *os);
    if (config->thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile(s).
std::vector<InputFile *> BitcodeCompiler::compile() {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);
  files.resize(maxTasks);

  // The --thinlto-cache-dir option specifies the path to a directory in which
  // to cache native object files for ThinLTO incremental builds. If a path was
  // specified, configure LTO to use it as the cache directory.
  lto::NativeObjectCache cache;
  if (!config->thinLTOCacheDir.empty())
    cache = check(
        lto::localCache(config->thinLTOCacheDir,
                        [&](size_t task, std::unique_ptr<MemoryBuffer> mb) {
                          files[task] = std::move(mb);
                        }));

  if (!bitcodeFiles.empty())
    checkError(ltoObj->run(
        [&](size_t task) {
          return std::make_unique<lto::NativeObjectStream>(
              std::make_unique<raw_svector_ostream>(buf[task]));
        },
        cache));

  // Emit empty index files for non-indexed files but not in single-module mode.
  if (config->thinLTOModulesToCompile.empty()) {
    for (StringRef s : thinIndices) {
      std::string path = getThinLTOOutputFile(s);
      openFile(path + ".thinlto.bc");
      if (config->thinLTOEmitImportsFiles)
        openFile(path + ".imports");
    }
  }

  if (config->thinLTOIndexOnly) {
    thinLTOCreateEmptyIndexFiles();

    if (!config->ltoObjPath.empty())
      saveBuffer(buf[0], config->ltoObjPath);

    // ThinLTO with index only option is required to generate only the index
    // files. After that, we exit from linker and ThinLTO backend runs in a
    // distributed environment.
    if (indexFile)
      indexFile->close();
    return {};
  }

  if (!config->thinLTOCacheDir.empty())
    pruneCache(config->thinLTOCacheDir, config->thinLTOCachePolicy);

  if (!config->ltoObjPath.empty()) {
    saveBuffer(buf[0], config->ltoObjPath);
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i], config->ltoObjPath + Twine(i));
  }

  if (config->saveTemps) {
    if (!buf[0].empty())
      saveBuffer(buf[0], config->outputFile + ".lto.o");
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i], config->outputFile + Twine(i) + ".lto.o");
  }

  if (config->ltoEmitAsm) {
    saveBuffer(buf[0], config->outputFile);
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i], config->outputFile + Twine(i));
    return {};
  }

  std::vector<InputFile *> ret;
  for (unsigned i = 0; i != maxTasks; ++i)
    if (!buf[i].empty())
      ret.push_back(createObjectFile(MemoryBufferRef(buf[i], "lto.tmp")));

  for (std::unique_ptr<MemoryBuffer> &file : files)
    if (file)
      ret.push_back(createObjectFile(*file));
  return ret;
}
