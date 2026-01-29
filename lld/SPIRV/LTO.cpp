//===- LTO.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LTO.h"
#include "Config.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/TargetOptionsCommandFlags.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace lld;
using namespace lld::spirv;

static lto::Config createConfig(Config &config) {
  lto::Config c;
  c.OptLevel = config.ltoOptLevel;
  c.DisableVerify = false;
  c.DiagHandler = diagnosticHandler;
  c.DefaultTriple = config.targetTriple;
  c.Options = lld::initTargetOptionsFromCodeGenFlags();
  c.RelocModel = Reloc::Static;
  c.CodeModel = getCodeModelFromCMModel();
  c.HasWholeProgramVisibility = true;
  c.CGFileType = CodeGenFileType::ObjectFile;
  c.OverrideTriple = config.targetTriple;
  return c;
}

BitcodeCompiler::BitcodeCompiler(Config &Config) : config(Config) {
  lto::ThinBackend backend = lto::createInProcessThinBackend(
      llvm::heavyweight_hardware_concurrency(1));
  ltoObj = std::make_unique<lto::LTO>(createConfig(config), backend, 1);
}

void BitcodeCompiler::add(MemoryBufferRef mb) {
  Expected<std::unique_ptr<lto::InputFile>> objOrErr =
      lto::InputFile::create(mb);
  if (!objOrErr) {
    error("failed to parse bitcode: " + toString(objOrErr.takeError()));
    return;
  }
  std::unique_ptr<lto::InputFile> &obj = *objOrErr;

  std::vector<lto::SymbolResolution> resolutions;
  resolutions.reserve(obj->symbols().size());
  for (const lto::InputFile::Symbol &sym : obj->symbols()) {
    lto::SymbolResolution res;
    res.Prevailing = !sym.isUndefined();
    res.FinalDefinitionInLinkageUnit = !sym.isUndefined();
    res.VisibleToRegularObj = true;
    res.ExportDynamic = false;
    res.LinkerRedefined = false;
    resolutions.push_back(res);
  }

  if (Error err = ltoObj->add(std::move(obj), std::move(resolutions)))
    error("failed to add bitcode: " + toString(std::move(err)));
}

std::vector<char> BitcodeCompiler::compile() {
  unsigned maxTasks = ltoObj->getMaxTasks();
  buf.resize(maxTasks);
  tempFiles.resize(maxTasks);

  // Run LTO to generate SPIR-V modules
  if (Error err = ltoObj->run([&](size_t task, const Twine &moduleName) {
        return std::make_unique<CachedFileStream>(
            std::make_unique<raw_svector_ostream>(buf[task]));
      })) {
    error("LTO failed: " + toString(std::move(err)));
    return {};
  }

  // Save SPIR-V modules to temp files for spirv-link
  std::vector<std::string> spirvFiles;
  for (unsigned i = 0; i < maxTasks; ++i) {
    if (buf[i].empty())
      continue;

    SmallString<128> tempPath;
    std::error_code ec = sys::fs::createTemporaryFile("lto", "spv", tempPath);
    if (ec) {
      error("failed to create temp file: " + ec.message());
      return {};
    }

    raw_fd_ostream os(tempPath, ec, sys::fs::OF_None);
    if (ec) {
      error("failed to open temp file: " + ec.message());
      return {};
    }
    os.write(buf[i].data(), buf[i].size());
    os.close();

    tempFiles[i] = tempPath.str().str();
    spirvFiles.push_back(tempFiles[i]);
  }

  if (spirvFiles.empty()) {
    error("LTO produced no output");
    return {};
  }

  // Use spirv-link to do the actual linking
  SmallString<128> outputPath;
  std::error_code ec =
      sys::fs::createTemporaryFile("linked", "spv", outputPath);
  if (ec) {
    error("failed to create output temp file: " + ec.message());
    return {};
  }
  ErrorOr<std::string> spirvLinkPath = sys::findProgramByName("spirv-link");
  if (!spirvLinkPath) {
    error("couldn't find path to spirv-link");
    return {};
  }

  std::vector<StringRef> linkArgs = {*spirvLinkPath};
  for (const std::string &f : spirvFiles)
    linkArgs.push_back(f);
  linkArgs.push_back("-o");
  linkArgs.push_back(outputPath);

  std::string errMsg;
  int result = sys::ExecuteAndWait(linkArgs[0], linkArgs, std::nullopt, {}, 0,
                                   0, &errMsg);
  if (result != 0) {
    error("spirv-link failed: " + errMsg);
    return {};
  }

  // Read linked output
  ErrorOr<std::unique_ptr<MemoryBuffer>> bufOrErr =
      MemoryBuffer::getFile(outputPath);
  if (auto ec = bufOrErr.getError()) {
    error("failed to read spirv-link output: " + ec.message());
    return {};
  }

  std::vector<char> linkedResult((*bufOrErr)->getBufferStart(),
                                 (*bufOrErr)->getBufferEnd());

  // Clean up all temp files
  sys::fs::remove(outputPath);
  for (const std::string &f : tempFiles)
    if (!f.empty())
      sys::fs::remove(f);

  return linkedResult;
}
