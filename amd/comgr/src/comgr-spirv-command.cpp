/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-spirv-command.h"

#ifndef COMGR_DISABLE_SPIRV
#include "comgr-diagnostic-handler.h"

#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
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

  if (!readSpirv(Context, ISS, M, Err)) {
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
