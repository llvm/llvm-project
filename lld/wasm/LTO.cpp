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
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
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
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace lld::wasm;
using namespace lld;

static lto::Config createConfig() {
  lto::Config c;
  c.Options = initTargetOptionsFromCodeGenFlags();

  // Always emit a section per function/data with LTO.
  c.Options.FunctionSections = true;
  c.Options.DataSections = true;

  c.DisableVerify = config->disableVerify;
  c.DiagHandler = diagnosticHandler;
  c.OptLevel = config->ltoo;
  c.MAttrs = getMAttrs();
  c.CGOptLevel = config->ltoCgo;
  c.DebugPassManager = config->ltoDebugPassManager;
  c.AlwaysEmitRegularLTOObj = !config->ltoObjPath.empty();

  if (config->relocatable)
    c.RelocModel = std::nullopt;
  else if (ctx.isPic)
    c.RelocModel = Reloc::PIC_;
  else
    c.RelocModel = Reloc::Static;

  if (config->saveTemps)
    checkError(c.addSaveTemps(config->outputFile.str() + ".",
                              /*UseInputModulePath*/ true));
  return c;
}

namespace lld::wasm {

BitcodeCompiler::BitcodeCompiler() {
  // Initialize indexFile.
  if (!config->thinLTOIndexOnlyArg.empty())
    indexFile = openFile(config->thinLTOIndexOnlyArg);

  // Initialize ltoObj.
  lto::ThinBackend backend;
  auto onIndexWrite = [&](StringRef s) { thinIndices.erase(s); };
  if (config->thinLTOIndexOnly) {
    backend = lto::createWriteIndexesThinBackend(
        llvm::hardware_concurrency(config->thinLTOJobs), "", "", "",
        config->thinLTOEmitImportsFiles, indexFile.get(), onIndexWrite);
  } else {
    backend = lto::createInProcessThinBackend(
        llvm::heavyweight_hardware_concurrency(config->thinLTOJobs),
        onIndexWrite, config->thinLTOEmitIndexFiles,
        config->thinLTOEmitImportsFiles);
  }
  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend,
                                      config->ltoPartitions);
}

BitcodeCompiler::~BitcodeCompiler() = default;

static void undefine(Symbol *s) {
  if (auto f = dyn_cast<DefinedFunction>(s))
    replaceSymbol<UndefinedFunction>(f, f->getName(), std::nullopt,
                                     std::nullopt, 0, f->getFile(),
                                     f->signature);
  else if (isa<DefinedData>(s))
    replaceSymbol<UndefinedData>(s, s->getName(), 0, s->getFile());
  else
    llvm_unreachable("unexpected symbol kind");
}

void BitcodeCompiler::add(BitcodeFile &f) {
  lto::InputFile &obj = *f.obj;
  unsigned symNum = 0;
  ArrayRef<Symbol *> syms = f.getSymbols();
  std::vector<lto::SymbolResolution> resols(syms.size());

  if (config->thinLTOEmitIndexFiles) {
    thinIndices.insert(obj.getName());
  }

  // Provide a resolution to the LTO API for each symbol.
  for (const lto::InputFile::Symbol &objSym : obj.symbols()) {
    Symbol *sym = syms[symNum];
    lto::SymbolResolution &r = resols[symNum];
    ++symNum;

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    r.Prevailing = !objSym.isUndefined() && sym->getFile() == &f;
    r.VisibleToRegularObj = config->relocatable || sym->isUsedInRegularObj ||
                            sym->isNoStrip() ||
                            (r.Prevailing && sym->isExported());
    if (r.Prevailing)
      undefine(sym);

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
  DenseSet<StringRef> linkedBitCodeFiles;
  for (BitcodeFile *f : ctx.bitcodeFiles)
    linkedBitCodeFiles.insert(f->getName());

  for (BitcodeFile *f : ctx.lazyBitcodeFiles) {
    if (!f->lazy)
      continue;
    if (linkedBitCodeFiles.contains(f->getName()))
      continue;
    std::string path(f->obj->getName());
    std::unique_ptr<raw_fd_ostream> os = openFile(path + ".thinlto.bc");
    if (!os)
      continue;

    ModuleSummaryIndex m(/*HaveGVs*/ false);
    m.setSkipModuleByDistributedBackend();
    writeIndexToFile(m, *os);
    if (config->thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting objects.
std::vector<StringRef> BitcodeCompiler::compile() {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);
  files.resize(maxTasks);

  // The --thinlto-cache-dir option specifies the path to a directory in which
  // to cache native object files for ThinLTO incremental builds. If a path was
  // specified, configure LTO to use it as the cache directory.
  FileCache cache;
  if (!config->thinLTOCacheDir.empty())
    cache = check(localCache("ThinLTO", "Thin", config->thinLTOCacheDir,
                             [&](size_t task, const Twine &moduleName,
                                 std::unique_ptr<MemoryBuffer> mb) {
                               files[task] = std::move(mb);
                             }));

  checkError(ltoObj->run(
      [&](size_t task, const Twine &moduleName) {
        buf[task].first = moduleName.str();
        return std::make_unique<CachedFileStream>(
            std::make_unique<raw_svector_ostream>(buf[task].second));
      },
      cache));

  // Emit empty index files for non-indexed files but not in single-module mode.
  for (StringRef s : thinIndices) {
    std::string path(s);
    openFile(path + ".thinlto.bc");
    if (config->thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }

  if (config->thinLTOEmitIndexFiles)
    thinLTOCreateEmptyIndexFiles();

  if (config->thinLTOIndexOnly) {
    if (!config->ltoObjPath.empty())
      saveBuffer(buf[0].second, config->ltoObjPath);

    // ThinLTO with index only option is required to generate only the index
    // files. After that, we exit from linker and ThinLTO backend runs in a
    // distributed environment.
    if (indexFile)
      indexFile->close();
    return {};
  }

  if (!config->thinLTOCacheDir.empty())
    pruneCache(config->thinLTOCacheDir, config->thinLTOCachePolicy, files);

  std::vector<StringRef> ret;
  for (unsigned i = 0; i != maxTasks; ++i) {
    StringRef objBuf = buf[i].second;
    StringRef bitcodeFilePath = buf[i].first;
    if (objBuf.empty())
      continue;
    ret.emplace_back(objBuf.data(), objBuf.size());
    if (!config->saveTemps)
      continue;

    // If the input bitcode file is path/to/x.o and -o specifies a.out, the
    // corresponding native relocatable file path will look like:
    // path/to/a.out.lto.x.o.
    StringRef ltoObjName;
    if (bitcodeFilePath == "ld-temp.o") {
      ltoObjName =
          saver().save(Twine(config->outputFile) + ".lto" +
                       (i == 0 ? Twine("") : Twine('.') + Twine(i)) + ".o");
    } else {
      StringRef directory = sys::path::parent_path(bitcodeFilePath);
      // For an archive member, which has an identifier like "d/a.a(coll.o at
      // 8)" (see BitcodeFile::BitcodeFile), use the filename; otherwise, use
      // the stem (d/a.o => a).
      StringRef baseName = bitcodeFilePath.ends_with(")")
                               ? sys::path::filename(bitcodeFilePath)
                               : sys::path::stem(bitcodeFilePath);
      StringRef outputFileBaseName = sys::path::filename(config->outputFile);
      SmallString<256> path;
      sys::path::append(path, directory,
                        outputFileBaseName + ".lto." + baseName + ".o");
      sys::path::remove_dots(path, true);
      ltoObjName = saver().save(path.str());
    }
    saveBuffer(objBuf, ltoObjName);
  }

  if (!config->ltoObjPath.empty()) {
    saveBuffer(buf[0].second, config->ltoObjPath);
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i].second, config->ltoObjPath + Twine(i));
  }

  for (std::unique_ptr<MemoryBuffer> &file : files)
    if (file)
      ret.push_back(file->getBuffer());

  return ret;
}

} // namespace lld::wasm
