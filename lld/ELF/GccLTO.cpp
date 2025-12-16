//===- GccLTO.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/config.h"

#if LLD_ENABLE_GNU_LTO
#include "Config.h"
#include "GccLTO.h"
#include "InputFiles.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

GccIRCompiler *gcc = nullptr;

GccIRCompiler::GccIRCompiler(Ctx &ctx) : IRCompiler(ctx) {
  initializeTv();
  assert(gcc == nullptr);
  gcc = this;
  loadPlugin();
}

GccIRCompiler::~GccIRCompiler() { gcc = nullptr; }

void GccIRCompiler::loadPlugin() {
  std::string Error;
  plugin = llvm::sys::DynamicLibrary::getPermanentLibrary(ctx.arg.plugin.data(),
                                                          &Error);
  if (!plugin.isValid()) {
    Err(ctx) << Error;
    return;
  }
  pluginOnLoad *onload = (pluginOnLoad *)plugin.getAddressOfSymbol("onload");
  if (!onload) {
    Err(ctx) << "Plugin does not provide onload()";
    return;
  }

  (*onload)(tv.data());
}

PluginStatus GccIRCompiler::registerClaimFile(pluginClaimFileHandler handler) {
  gcc->claimFileHandler = handler;
  return LDPS_OK;
}

PluginStatus
GccIRCompiler::registerClaimFileV2(pluginClaimFileHandlerV2 handler) {
  gcc->claimFileHandlerV2 = handler;
  return LDPS_OK;
}

PluginStatus regAllSymbolsRead(pluginAllSymbolsReadHandler handler) {
  return gcc->registerAllSymbolsRead(handler);
}

PluginStatus
GccIRCompiler::registerAllSymbolsRead(pluginAllSymbolsReadHandler handler) {
  allSymbolsReadHandler = handler;
  return LDPS_OK;
}

static PluginStatus addSymbols(void *handle, int nsyms,
                               const struct PluginSymbol *syms) {
  ELFFileBase *f = (ELFFileBase *)handle;
  if (f == NULL)
    return LDPS_ERR;

  for (int i = 0; i < nsyms; i++) {
    // TODO: Add symbols.
    // TODO: Convert these symbosl into ArrayRef<lto::InputFile::Symbol> and
    // ArrayRef<Symbol *> ?
  }

  return LDPS_OK;
}

static PluginStatus getSymbols(const void *handle, int nsyms,
                               struct PluginSymbol *syms) {
  for (int i = 0; i < nsyms; i++) {
    syms[i].resolution = LDPR_UNDEF;
    // TODO: Implement other scenarios.
  }
  return LDPS_OK;
}

PluginStatus addInputFile(const char *pathname) {
  if (gcc->addCompiledFile(StringRef(pathname)))
    return LDPS_OK;
  else
    return LDPS_ERR;
}

void GccIRCompiler::initializeTv() {
  tv.push_back(PluginTV(LDPT_MESSAGE, &message));
  tv.push_back(PluginTV(LDPT_API_VERSION, LD_PLUGIN_API_VERSION));
  for (std::string &s : ctx.arg.pluginOpt)
    tv.push_back(PluginTV(LDPT_OPTION, s.c_str()));
  PluginFileType o;
  if (ctx.arg.pie)
    o = LDPO_PIE;
  else if (ctx.arg.relocatable)
    o = LDPO_REL;
  else if (ctx.arg.shared)
    o = LDPO_DYN;
  else
    o = LDPO_EXEC;
  tv.push_back(PluginTV(LDPT_LINKER_OUTPUT, o));
  tv.push_back(PluginTV(LDPT_OUTPUT_NAME, ctx.arg.outputFile.data()));
  // Share the address of a C wrapper that is API-compatible with
  // plugin-api.h.
  tv.push_back(PluginTV(LDPT_REGISTER_CLAIM_FILE_HOOK, &registerClaimFile));
  tv.push_back(
      PluginTV(LDPT_REGISTER_CLAIM_FILE_HOOK_V2, &registerClaimFileV2));

  tv.push_back(PluginTV(LDPT_ADD_SYMBOLS, &addSymbols));
  tv.push_back(
      PluginTV(LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK, &regAllSymbolsRead));
  tv.push_back(PluginTV(LDPT_GET_SYMBOLS, &getSymbols));
  tv.push_back(PluginTV(LDPT_ADD_INPUT_FILE, &addInputFile));
}

void GccIRCompiler::add(ELFFileBase &f) {
  PluginInputFile file;

  std::string name = f.getName().str();
  file.name = f.getName().data();
  file.handle = const_cast<void *>(reinterpret_cast<const void *>(&f));

  std::error_code ec = sys::fs::openFileForRead(name, file.fd);
  if (ec) {
    Err(ctx) << "Cannot open file " + name + ": " + ec.message();
    return;
  }
  file.offset = 0;
  uint64_t size;
  ec = sys::fs::file_size(name, size);
  if (ec) {
    Err(ctx) << "Cannot get the size of file " + name + ": " + ec.message();
    sys::fs::closeFile(file.fd);
    return;
  }
  if (size > 0 && size <= INT_MAX)
    file.filesize = size;

  int claimed;
  PluginStatus status = claimFileHandlerV2(&file, &claimed, 1);

  if (status != LDPS_OK)
    Err(ctx) << "liblto returned " + std::to_string(status);

  ec = sys::fs::closeFile(file.fd);
  if (ec) {
    Err(ctx) << ec.message();
  }
}

SmallVector<std::unique_ptr<InputFile>, 0> GccIRCompiler::compile() {
  SmallVector<std::unique_ptr<InputFile>, 0> ret;
  PluginStatus status = allSymbolsReadHandler();
  if (status != LDPS_OK)
    Err(ctx) << "The plugin returned an error after all symbols were read.";

  for (auto &m : files) {
    ret.push_back(createObjFile(ctx, m->getMemBufferRef()));
  }
  return ret;
}

void GccIRCompiler::addObject(IRFile &f,
                              std::vector<llvm::lto::SymbolResolution> &r) {
  // TODO: Implement this.
}

enum PluginStatus GccIRCompiler::message(int level, const char *format, ...) {
  // TODO: Implement this function.
  return LDPS_OK;
}

bool GccIRCompiler::addCompiledFile(StringRef path) {
  std::optional<MemoryBufferRef> mbref = readFile(ctx, path);
  if (!mbref)
    return false;
  files.push_back(MemoryBuffer::getMemBuffer(*mbref));
  return true;
}
#endif
