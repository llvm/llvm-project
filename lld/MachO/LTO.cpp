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

#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;
using namespace lld::macho;
using namespace llvm;

static lto::Config createConfig() {
  lto::Config c;
  c.Options = initTargetOptionsFromCodeGenFlags();
  return c;
}

BitcodeCompiler::BitcodeCompiler() {
  auto backend =
      lto::createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());
  ltoObj = std::make_unique<lto::LTO>(createConfig(), backend);
}

void BitcodeCompiler::add(BitcodeFile &f) {
  ArrayRef<lto::InputFile::Symbol> objSyms = f.obj->symbols();
  std::vector<lto::SymbolResolution> resols;
  resols.reserve(objSyms.size());

  // Provide a resolution to the LTO API for each symbol.
  for (const lto::InputFile::Symbol &objSym : objSyms) {
    resols.emplace_back();
    lto::SymbolResolution &r = resols.back();

    // Ideally we shouldn't check for SF_Undefined but currently IRObjectFile
    // reports two symbols for module ASM defined. Without this check, lld
    // flags an undefined in IR with a definition in ASM as prevailing.
    // Once IRObjectFile is fixed to report only one symbol this hack can
    // be removed.
    r.Prevailing = !objSym.isUndefined();

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

  // TODO: set modTime properly
  std::vector<ObjFile *> ret;
  for (unsigned i = 0; i != maxTasks; ++i)
    if (!buf[i].empty())
      ret.push_back(
          make<ObjFile>(MemoryBufferRef(buf[i], "lto.tmp"), /*modTime=*/0, ""));

  return ret;
}
