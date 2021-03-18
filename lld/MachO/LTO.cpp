//===- LTO.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LTO.h"
#include "Config.h"
#include "Driver.h"
#include "InputFiles.h"
#include "Symbols.h"

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/ObjCARC.h"

using namespace lld;
using namespace lld::macho;
using namespace llvm;
using namespace llvm::sys;

static lto::Config createConfig() {
  lto::Config c;
  c.Options = initTargetOptionsFromCodeGenFlags();
  c.CodeModel = getCodeModelFromCMModel();
  c.CPU = getCPUStr();
  c.MAttrs = getMAttrs();
  c.UseNewPM = config->ltoNewPassManager;
  c.PreCodeGenPassesHook = [](legacy::PassManager &pm) {
    pm.add(createObjCARCContractPass());
  };
  return c;
}

BitcodeCompiler::BitcodeCompiler() {
  lto::ThinBackend backend =
      lto::createInProcessThinBackend(heavyweight_hardware_concurrency());
  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend);
}

void BitcodeCompiler::add(BitcodeFile &f) {
  ArrayRef<lto::InputFile::Symbol> objSyms = f.obj->symbols();
  std::vector<lto::SymbolResolution> resols;
  resols.reserve(objSyms.size());

  // Provide a resolution to the LTO API for each symbol.
  auto symIt = f.symbols.begin();
  for (const lto::InputFile::Symbol &objSym : objSyms) {
    resols.emplace_back();
    lto::SymbolResolution &r = resols.back();
    Symbol *sym = *symIt++;

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    r.Prevailing = !objSym.isUndefined() && sym->getFile() == &f;

    // Un-define the symbol so that we don't get duplicate symbol errors when we
    // load the ObjFile emitted by LTO compilation.
    if (r.Prevailing)
      replaceSymbol<Undefined>(sym, sym->getName(), sym->getFile(),
                               RefState::Strong);

    // TODO: set the other resolution configs properly
    r.VisibleToRegularObj = true;
  }
  checkError(ltoObj->add(std::move(f.obj), resols));
}

// Merge all the bitcode files we have seen, codegen the result
// and return the resulting ObjectFile(s).
std::vector<ObjFile *> BitcodeCompiler::compile() {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);

  checkError(ltoObj->run([&](size_t task) {
    return std::make_unique<lto::NativeObjectStream>(
        std::make_unique<raw_svector_ostream>(buf[task]));
  }));

  if (config->saveTemps) {
    if (!buf[0].empty())
      saveBuffer(buf[0], config->outputFile + ".lto.o");
    for (unsigned i = 1; i != maxTasks; ++i)
      saveBuffer(buf[i], config->outputFile + Twine(i) + ".lto.o");
  }

  if (!config->ltoObjPath.empty())
    fs::create_directories(config->ltoObjPath);

  std::vector<ObjFile *> ret;
  for (unsigned i = 0; i != maxTasks; ++i) {
    if (buf[i].empty())
      continue;
    SmallString<261> filePath("/tmp/lto.tmp");
    uint32_t modTime = 0;
    if (!config->ltoObjPath.empty()) {
      filePath = config->ltoObjPath;
      path::append(filePath, Twine(i) + "." +
                                 getArchitectureName(config->target.Arch) +
                                 ".lto.o");
      saveBuffer(buf[i], filePath);
      modTime = getModTime(filePath);
    }
    ret.push_back(make<ObjFile>(
        MemoryBufferRef(buf[i], saver.save(filePath.str())), modTime, ""));
  }

  return ret;
}
