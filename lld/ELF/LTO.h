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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <plugin-api.h>
#include <vector>

namespace llvm::lto {
class LTO;
class SymbolResolution;
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
  void add(IRFile &f);
  virtual SmallVector<std::unique_ptr<InputFile>, 0> compile() = 0;
};

class BitcodeCompiler : public IRCompiler {
protected:
  void addObject(IRFile &f,
                 std::vector<llvm::lto::SymbolResolution> &r) override;

public:
  BitcodeCompiler(Ctx &ctx);
  ~BitcodeCompiler();

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

class GccIRCompiler : public IRCompiler {
protected:
  void addObject(IRFile &f,
                 std::vector<llvm::lto::SymbolResolution> &r) override;

public:
  ~GccIRCompiler();
  static GccIRCompiler *getInstance();
  static GccIRCompiler *getInstance(Ctx &ctx);

  SmallVector<std::unique_ptr<InputFile>, 0> compile() override;
  static enum ld_plugin_status message(int level, const char *format, ...);

private:
  GccIRCompiler(Ctx &ctx);
  static GccIRCompiler *singleton;
  struct ld_plugin_tv *tv;
  // Handle for the shared library created via dlopen().
  void *plugin;

  void initializeTv();
};

} // namespace lld::elf

#endif
