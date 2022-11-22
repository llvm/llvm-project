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
#include "Symbols.h"
#include "lld/Common/Args.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/CodeGen.h"
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
using namespace lld;
using namespace lld::coff;

// Creates an empty file to and returns a raw_fd_ostream to write to it.
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

static std::string getThinLTOOutputFile(StringRef path) {
  return lto::getThinLTOOutputFile(
      std::string(path), std::string(config->thinLTOPrefixReplace.first),
      std::string(config->thinLTOPrefixReplace.second));
}

static lto::Config createConfig() {
  lto::Config c;
  c.Options = initTargetOptionsFromCodeGenFlags();
  c.Options.EmitAddrsig = true;

  // Always emit a section per function/datum with LTO. LLVM LTO should get most
  // of the benefit of linker GC, but there are still opportunities for ICF.
  c.Options.FunctionSections = true;
  c.Options.DataSections = true;

  // Use static reloc model on 32-bit x86 because it usually results in more
  // compact code, and because there are also known code generation bugs when
  // using the PIC model (see PR34306).
  if (config->machine == COFF::IMAGE_FILE_MACHINE_I386)
    c.RelocModel = Reloc::Static;
  else
    c.RelocModel = Reloc::PIC_;
#ifndef NDEBUG
  c.DisableVerify = false;
#else
  c.DisableVerify = true;
#endif
  c.DiagHandler = diagnosticHandler;
  c.OptLevel = config->ltoo;
  c.CPU = getCPUStr();
  c.MAttrs = getMAttrs();
  c.CGOptLevel = args::getCGOptLevel(config->ltoo);
  c.AlwaysEmitRegularLTOObj = !config->ltoObjPath.empty();
  c.DebugPassManager = config->ltoDebugPassManager;
  c.CSIRProfile = std::string(config->ltoCSProfileFile);
  c.RunCSIRInstr = config->ltoCSProfileGenerate;
  c.PGOWarnMismatch = config->ltoPGOWarnMismatch;

  if (config->saveTemps)
    checkError(c.addSaveTemps(std::string(config->outputFile) + ".",
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
    auto OnIndexWrite = [&](StringRef S) { thinIndices.erase(S); };
    backend = lto::createWriteIndexesThinBackend(
        std::string(config->thinLTOPrefixReplace.first),
        std::string(config->thinLTOPrefixReplace.second),
        config->thinLTOEmitImportsFiles, indexFile.get(), OnIndexWrite);
  } else {
    backend = lto::createInProcessThinBackend(
        llvm::heavyweight_hardware_concurrency(config->thinLTOJobs));
  }

  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend,
                                       config->ltoPartitions);
}

BitcodeCompiler::~BitcodeCompiler() = default;

static void undefine(Symbol *s) { replaceSymbol<Undefined>(s, s->getName()); }

void BitcodeCompiler::add(BitcodeFile &f) {
  lto::InputFile &obj = *f.obj;
  unsigned symNum = 0;
  std::vector<Symbol *> symBodies = f.getSymbols();
  std::vector<lto::SymbolResolution> resols(symBodies.size());

  if (config->thinLTOIndexOnly)
    thinIndices.insert(obj.getName());

  // Provide a resolution to the LTO API for each symbol.
  for (const lto::InputFile::Symbol &objSym : obj.symbols()) {
    Symbol *sym = symBodies[symNum];
    lto::SymbolResolution &r = resols[symNum];
    ++symNum;

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    r.Prevailing = !objSym.isUndefined() && sym->getFile() == &f;
    r.VisibleToRegularObj = sym->isUsedInRegularObj;
    if (r.Prevailing)
      undefine(sym);

    // We tell LTO to not apply interprocedural optimization for wrapped
    // (with -wrap) symbols because otherwise LTO would inline them while
    // their values are still not final.
    r.LinkerRedefined = !sym->canInline;
  }
  checkError(ltoObj->add(std::move(f.obj), resols));
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting objects.
std::vector<InputFile *> BitcodeCompiler::compile(COFFLinkerContext &ctx) {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);
  files.resize(maxTasks);

  // The /lldltocache option specifies the path to a directory in which to cache
  // native object files for ThinLTO incremental builds. If a path was
  // specified, configure LTO to use it as the cache directory.
  FileCache cache;
  if (!config->ltoCache.empty())
    cache =
        check(localCache("ThinLTO", "Thin", config->ltoCache,
                         [&](size_t task, std::unique_ptr<MemoryBuffer> mb) {
                           files[task] = std::move(mb);
                         }));

  checkError(ltoObj->run(
      [&](size_t task) {
        return std::make_unique<CachedFileStream>(
            std::make_unique<raw_svector_ostream>(buf[task]));
      },
      cache));

  // Emit empty index files for non-indexed files
  for (StringRef s : thinIndices) {
    std::string path = getThinLTOOutputFile(s);
    openFile(path + ".thinlto.bc");
    if (config->thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }

  // ThinLTO with index only option is required to generate only the index
  // files. After that, we exit from linker and ThinLTO backend runs in a
  // distributed environment.
  if (config->thinLTOIndexOnly) {
    if (!config->ltoObjPath.empty())
      saveBuffer(buf[0], config->ltoObjPath);
    if (indexFile)
      indexFile->close();
    return {};
  }

  if (!config->ltoCache.empty())
    pruneCache(config->ltoCache, config->ltoCachePolicy, files);

  std::vector<InputFile *> ret;
  for (unsigned i = 0; i != maxTasks; ++i) {
    // Assign unique names to LTO objects. This ensures they have unique names
    // in the PDB if one is produced. The names should look like:
    // - foo.exe.lto.obj
    // - foo.exe.lto.1.obj
    // - ...
    StringRef ltoObjName =
        saver().save(Twine(config->outputFile) + ".lto" +
                     (i == 0 ? Twine("") : Twine('.') + Twine(i)) + ".obj");

    // Get the native object contents either from the cache or from memory.  Do
    // not use the cached MemoryBuffer directly, or the PDB will not be
    // deterministic.
    StringRef objBuf;
    if (files[i])
      objBuf = files[i]->getBuffer();
    else
      objBuf = buf[i];
    if (objBuf.empty())
      continue;

    if (config->saveTemps)
      saveBuffer(buf[i], ltoObjName);
    ret.push_back(make<ObjFile>(ctx, MemoryBufferRef(objBuf, ltoObjName)));
  }

  return ret;
}
