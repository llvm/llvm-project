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
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace lld::wasm;
using namespace lld;

static std::string getThinLTOOutputFile(StringRef modulePath) {
  return lto::getThinLTOOutputFile(modulePath, ctx.arg.thinLTOPrefixReplaceOld,
                                   ctx.arg.thinLTOPrefixReplaceNew);
}

static lto::Config createConfig() {
  lto::Config c;
  c.Options = initTargetOptionsFromCodeGenFlags();

  // Always emit a section per function/data with LTO.
  c.Options.FunctionSections = true;
  c.Options.DataSections = true;

  c.DisableVerify = ctx.arg.disableVerify;
  c.DiagHandler = diagnosticHandler;
  c.OptLevel = ctx.arg.ltoo;
  c.CPU = getCPUStr();
  c.MAttrs = getMAttrs();
  c.CGOptLevel = ctx.arg.ltoCgo;
  c.DebugPassManager = ctx.arg.ltoDebugPassManager;
  c.AlwaysEmitRegularLTOObj = !ctx.arg.ltoObjPath.empty();

  if (ctx.arg.relocatable)
    c.RelocModel = std::nullopt;
  else if (ctx.isPic)
    c.RelocModel = Reloc::PIC_;
  else
    c.RelocModel = Reloc::Static;

  if (ctx.arg.saveTemps)
    checkError(c.addSaveTemps(ctx.arg.outputFile.str() + ".",
                              /*UseInputModulePath*/ true));
  return c;
}

namespace lld::wasm {

BitcodeCompiler::BitcodeCompiler() {
  // Initialize indexFile.
  if (!ctx.arg.thinLTOIndexOnlyArg.empty())
    indexFile = openFile(ctx.arg.thinLTOIndexOnlyArg);

  // Initialize ltoObj.
  lto::ThinBackend backend;
  auto onIndexWrite = [&](StringRef s) { thinIndices.erase(s); };
  if (ctx.arg.thinLTOIndexOnly) {
    backend = lto::createWriteIndexesThinBackend(
        llvm::hardware_concurrency(ctx.arg.thinLTOJobs),
        std::string(ctx.arg.thinLTOPrefixReplaceOld),
        std::string(ctx.arg.thinLTOPrefixReplaceNew),
        std::string(ctx.arg.thinLTOPrefixReplaceNativeObject),
        ctx.arg.thinLTOEmitImportsFiles, indexFile.get(), onIndexWrite);
  } else {
    backend = lto::createInProcessThinBackend(
        llvm::heavyweight_hardware_concurrency(ctx.arg.thinLTOJobs),
        onIndexWrite, ctx.arg.thinLTOEmitIndexFiles,
        ctx.arg.thinLTOEmitImportsFiles);
  }
  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend,
                                      ctx.arg.ltoPartitions);
}

BitcodeCompiler::~BitcodeCompiler() = default;

static void undefine(Symbol *s) {
  if (auto f = dyn_cast<DefinedFunction>(s))
    // If the signature is null, there were no calls from non-bitcode objects.
    replaceSymbol<UndefinedFunction>(f, f->getName(), std::nullopt,
                                     std::nullopt, 0, f->getFile(),
                                     f->signature, f->signature != nullptr);
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

  if (ctx.arg.thinLTOEmitIndexFiles) {
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
    r.VisibleToRegularObj = ctx.arg.relocatable || sym->isUsedInRegularObj ||
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
    std::string path =
        replaceThinLTOSuffix(getThinLTOOutputFile(f->obj->getName()));
    std::unique_ptr<raw_fd_ostream> os = openFile(path + ".thinlto.bc");
    if (!os)
      continue;

    ModuleSummaryIndex m(/*HaveGVs*/ false);
    m.setSkipModuleByDistributedBackend();
    writeIndexToFile(m, *os);
    if (ctx.arg.thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting objects.
SmallVector<InputFile *, 0> BitcodeCompiler::compile() {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);
  files.resize(maxTasks);
  filenames.resize(maxTasks);

  // The --thinlto-cache-dir option specifies the path to a directory in which
  // to cache native object files for ThinLTO incremental builds. If a path was
  // specified, configure LTO to use it as the cache directory.
  FileCache cache;
  if (!ctx.arg.thinLTOCacheDir.empty())
    cache = check(localCache("ThinLTO", "Thin", ctx.arg.thinLTOCacheDir,
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
    if (ctx.arg.thinLTOEmitImportsFiles)
      openFile(path + ".imports");
  }

  if (ctx.arg.thinLTOEmitIndexFiles)
    thinLTOCreateEmptyIndexFiles();

  if (ctx.arg.thinLTOIndexOnly) {
    if (!ctx.arg.ltoObjPath.empty())
      saveBuffer(buf[0].second, ctx.arg.ltoObjPath);

    // ThinLTO with index only option is required to generate only the index
    // files. After that, we exit from linker and ThinLTO backend runs in a
    // distributed environment.
    if (indexFile)
      indexFile->close();
    return {};
  }

  if (!ctx.arg.thinLTOCacheDir.empty())
    pruneCache(ctx.arg.thinLTOCacheDir, ctx.arg.thinLTOCachePolicy, files);

  SmallVector<InputFile *, 0> ret;
  for (unsigned i = 0; i != maxTasks; ++i) {
    StringRef objBuf = buf[i].second;
    StringRef bitcodeFilePath = buf[i].first;
    if (files[i]) {
      // When files[i] is not null, we get the native relocatable file from the
      // cache. filenames[i] contains the original BitcodeFile's identifier.
      objBuf = files[i]->getBuffer();
      bitcodeFilePath = filenames[i];
    } else {
      objBuf = buf[i].second;
      bitcodeFilePath = buf[i].first;
    }
    if (objBuf.empty())
      continue;

    // If the input bitcode file is path/to/x.o and -o specifies a.out, the
    // corresponding native relocatable file path will look like:
    // path/to/a.out.lto.x.o.
    StringRef ltoObjName;
    if (bitcodeFilePath == "ld-temp.o") {
      ltoObjName =
          saver().save(Twine(ctx.arg.outputFile) + ".lto" +
                       (i == 0 ? Twine("") : Twine('.') + Twine(i)) + ".o");
    } else {
      StringRef directory = sys::path::parent_path(bitcodeFilePath);
      // For an archive member, which has an identifier like "d/a.a(coll.o at
      // 8)" (see BitcodeFile::BitcodeFile), use the filename; otherwise, use
      // the stem (d/a.o => a).
      StringRef baseName = bitcodeFilePath.ends_with(")")
                               ? sys::path::filename(bitcodeFilePath)
                               : sys::path::stem(bitcodeFilePath);
      StringRef outputFileBaseName = sys::path::filename(ctx.arg.outputFile);
      SmallString<256> path;
      sys::path::append(path, directory,
                        outputFileBaseName + ".lto." + baseName + ".o");
      sys::path::remove_dots(path, true);
      ltoObjName = saver().save(path.str());
    }
    if (ctx.arg.saveTemps)
      saveBuffer(objBuf, ltoObjName);
    ret.emplace_back(createObjectFile(MemoryBufferRef(objBuf, ltoObjName)));
  }

  if (!ctx.arg.ltoObjPath.empty()) {
    saveBuffer(buf[0].second, ctx.arg.ltoObjPath);
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i].second, ctx.arg.ltoObjPath + Twine(i));
  }

  return ret;
}

} // namespace lld::wasm
