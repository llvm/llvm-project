//===- comgr-symbolizer.cpp - Symbolizer implementation -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the amd_comgr_symbolize() API, leveraging LLVM's
/// LLVMSymbolizer class and llvm::symbolize namespace.
///
//===----------------------------------------------------------------------===//

#include "comgr-symbolizer.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace COMGR;

namespace {
// llvm symbolizer with default options
LLVMSymbolizer::Options getDefaultOptions() {
  LLVMSymbolizer::Options Opt;
  Opt.SkipLineZero = true;
  return Opt;
}

llvm::symbolize::PrinterConfig getDefaultPrinterConfig() {
  llvm::symbolize::PrinterConfig Config;
  Config.Pretty = true;
  Config.Verbose = false;
  Config.PrintFunctions = true;
  Config.PrintAddress = false;
  Config.SourceContextLines = 0;
  return Config;
}

llvm::symbolize::ErrorHandler
symbolizeErrorHandler(llvm::raw_string_ostream &OS) {
  return
      [&](const llvm::ErrorInfoBase &ErrorInfo, llvm::StringRef ErrorBanner) {
        OS << ErrorBanner;
        ErrorInfo.log(OS);
        OS << '\n';
      };
}
} // namespace

Symbolizer::Symbolizer(std::unique_ptr<ObjectFile> &&CodeObject,
                       PrintSymbolCallback PrintSymbol)
    : CodeObject(std::move(CodeObject)), PrintSymbol(PrintSymbol) {
  SymbolizerImpl = std::make_unique<LLVMSymbolizer>(getDefaultOptions());
}
Symbolizer::~Symbolizer() = default;

amd_comgr_status_t
Symbolizer::create(DataObject *CodeObjectP, PrintSymbolCallback PrintSymbol,
                   amd_comgr_symbolizer_info_t *SymbolizeInfo) {
  std::unique_ptr<llvm::MemoryBuffer> Buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(CodeObjectP->Data, CodeObjectP->Size), "", false);

  if (!Buf) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  auto ObjectOrErr = ObjectFile::createObjectFile(*Buf);
  if (errorToBool(ObjectOrErr.takeError())) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  std::unique_ptr<ObjectFile> ObjFile = std::move(ObjectOrErr.get());
  Symbolizer *SI =
      new (std::nothrow) Symbolizer(std::move(ObjFile), PrintSymbol);
  if (!SI) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *SymbolizeInfo = Symbolizer::convert(SI);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t Symbolizer::symbolize(uint64_t Address, bool IsCode,
                                         void *UserData) {

  std::string Result;
  llvm::raw_string_ostream OS(Result);
  llvm::symbolize::PrinterConfig Config = getDefaultPrinterConfig();
  llvm::symbolize::Request Request{"", Address, ""};
  auto Printer = std::make_unique<llvm::symbolize::LLVMPrinter>(
      OS, symbolizeErrorHandler(OS), Config);
  if (IsCode) {
    auto ResOrErr = SymbolizerImpl->symbolizeInlinedCode(
        *CodeObject, {Address, llvm::object::SectionedAddress::UndefSection});
    Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DIInliningInfo());
  } else { // data
    auto ResOrErr = SymbolizerImpl->symbolizeData(
        *CodeObject, {Address, llvm::object::SectionedAddress::UndefSection});
    Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DIGlobal());
  }

  PrintSymbol(Result.c_str(), UserData);
  return AMD_COMGR_STATUS_SUCCESS;
}
