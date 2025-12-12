//===- LTO.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a way to combine bitcode files into one ELF
// file by compiling them using LLVM.
//
// If LTO is in use, your input files are not in regular ELF files
// but instead LLVM bitcode files. In that case, the linker has to
// convert bitcode files into the native format so that we can create
// an ELF file that contains native code. This file provides that
// functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_LTO_H
#define LLD_ELF_LTO_H

#include "lld/Common/LLVM.h"
#include "lld/config.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#if LLD_ENABLE_GNU_LTO
#include "GnuLTO.h"
#endif
#include <memory>
#include <vector>

namespace llvm::lto {
class LTO;
struct SymbolResolution;
}

namespace lld::elf {
struct Ctx;
class BitcodeFile;
class ELFFileBase;
class InputFile;
class IRFile;
class BinaryFile;

class IRCompiler {
protected:
  Ctx &ctx;
  llvm::DenseSet<StringRef> thinIndices;
  llvm::DenseSet<StringRef> usedStartStop;
  virtual void addObject(IRFile &f,
                         std::vector<llvm::lto::SymbolResolution> &r) = 0;

public:
  IRCompiler(Ctx &ctx) : ctx(ctx) {}
  virtual ~IRCompiler() {}
  void add(IRFile &f);
  virtual SmallVector<std::unique_ptr<InputFile>, 0> compile() = 0;
};

class BitcodeCompiler : public IRCompiler {
protected:
  void addObject(IRFile &f,
                 std::vector<llvm::lto::SymbolResolution> &r) override;

public:
  BitcodeCompiler(Ctx &ctx);
  ~BitcodeCompiler() {}

  void add(BinaryFile &f);
  SmallVector<std::unique_ptr<InputFile>, 0> compile() override;

private:
  std::unique_ptr<llvm::lto::LTO> ltoObj;
  // An array of (module name, native relocatable file content) pairs.
  SmallVector<std::pair<std::string, SmallString<0>>, 0> buf;
  std::vector<std::unique_ptr<MemoryBuffer>> files;
  SmallVector<std::string, 0> filenames;
  std::unique_ptr<llvm::raw_fd_ostream> indexFile;
};

#if LLD_ENABLE_GNU_LTO
class GccIRCompiler : public IRCompiler {
protected:
  void addObject(IRFile &f,
                 std::vector<llvm::lto::SymbolResolution> &r) override;

public:
  GccIRCompiler(Ctx &ctx);
  ~GccIRCompiler();

  void add(ELFFileBase &f);
  SmallVector<std::unique_ptr<InputFile>, 0> compile() override;
  static PluginStatus message(int level, const char *format, ...);
  static PluginStatus registerClaimFile(pluginClaimFileHandler handler);
  static PluginStatus registerClaimFileV2(pluginClaimFileHandlerV2 handler);
  PluginStatus registerAllSymbolsRead(pluginAllSymbolsReadHandler handler);
  void loadPlugin();
  bool addCompiledFile(StringRef path);

private:
  std::vector<std::unique_ptr<MemoryBuffer>> files;
  SmallVector<PluginTV> tv;
  pluginClaimFileHandler *claimFileHandler;
  pluginClaimFileHandlerV2 *claimFileHandlerV2;
  pluginAllSymbolsReadHandler *allSymbolsReadHandler;
  // Handle for the shared library created via dlopen().
  llvm::sys::DynamicLibrary plugin;

  void initializeTv();
};
#endif

} // namespace lld::elf

#endif
