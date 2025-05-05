//===- comgr-spirv-command.cpp - SPIRVCommand implementation --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for the SPIRV to LLVM
/// Bitcode conversion.
///
//===----------------------------------------------------------------------===//

#include "comgr-spirv-command.h"

#ifndef COMGR_DISABLE_SPIRV
#include "comgr-diagnostic-handler.h"

#include <LLVMSPIRVLib.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>

#include <sstream>
#endif

namespace COMGR {
using namespace llvm;
Error SPIRVCommand::writeExecuteOutput(StringRef CachedBuffer) {
  assert(OutputBuffer.empty());
  OutputBuffer.reserve(CachedBuffer.size());
  OutputBuffer.insert(OutputBuffer.end(), CachedBuffer.begin(),
                      CachedBuffer.end());
  return Error::success();
}

Expected<StringRef> SPIRVCommand::readExecuteOutput() {
  return StringRef(OutputBuffer.data(), OutputBuffer.size());
}

amd_comgr_status_t SPIRVCommand::execute(raw_ostream &LogS) {
#ifndef COMGR_DISABLE_SPIRV
  LLVMContext Context;
  Context.setDiagnosticHandler(
      std::make_unique<AMDGPUCompilerDiagnosticHandler>(LogS), true);

  // TODO: With C++23, we should investigate replacing with spanstream
  // to avoid memory copies:
  //  https://en.cppreference.com/w/cpp/io/basic_ispanstream
  std::istringstream ISS(std::string(InputBuffer.data(), InputBuffer.size()));

  Module *M;
  std::string Err;

  SPIRV::TranslatorOpts Opts;
  Opts.enableAllExtensions();
  Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL20);

  if (!readSpirv(Context, Opts, ISS, M, Err)) {
    LogS << "Failed to load SPIR-V as LLVM Module: " << Err << '\n';
    return AMD_COMGR_STATUS_ERROR;
  }

  BitcodeWriter Writer(OutputBuffer);
  Writer.writeModule(*M, false, nullptr, false, nullptr);
  Writer.writeSymtab();
  Writer.writeStrtab();
  return AMD_COMGR_STATUS_SUCCESS;
#else
  return AMD_COMGR_STATUS_ERROR;
#endif
}

SPIRVCommand::ActionClass SPIRVCommand::getClass() const {
  // return an action class that is not allocated to distinguish it from any
  // clang action
  return clang::driver::Action::ActionClass::JobClassLast + 1;
}

void SPIRVCommand::addOptionsIdentifier(HashAlgorithm &) const {
  // do nothing, there are no options
  return;
}

Error SPIRVCommand::addInputIdentifier(HashAlgorithm &H) const {
  addString(H, InputBuffer);
  return Error::success();
}
} // namespace COMGR
