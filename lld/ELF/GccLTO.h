//===- GccLTO.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a way to combine GCC IR files into one ELF
// file by compiling them using GCC.
//
// If LTO is in use, your input files are not in regular ELF files
// but instead GCC IR files. In that case, the linker has to
// convert them into the native format so that we can create
// an ELF file that contains native code. This file provides that
// functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_GCCLTO_H
#define LLD_ELF_GCCLTO_H

#include "lld/config.h"
#if LLD_ENABLE_GNU_LTO
#include "GnuLTO.h"
#include "LTO.h"
#include "lld/Common/LLVM.h"
#include "llvm/Support/DynamicLibrary.h"

namespace lld::elf {
struct Ctx;

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
} // namespace lld::elf

#else
#error "LLD_ENABLE_GNU_LTO must be enabled before including GccLTO.h"
#endif
#endif // LLD_ELF_GCCLTO_H
