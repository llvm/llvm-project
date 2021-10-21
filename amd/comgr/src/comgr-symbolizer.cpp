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

#include "comgr-symbolizer.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace COMGR;

static llvm::symbolize::PrinterConfig getDefaultPrinterConfig() {
  llvm::symbolize::PrinterConfig Config;
  Config.Pretty = true;
  Config.Verbose = false;
  Config.PrintFunctions = true;
  Config.PrintAddress = false;
  Config.SourceContextLines = 0;
  return Config;
}

amd_comgr_status_t
Symbolizer::create(DataObject *CodeObjectP, PrintSymbolCallback PrintSymbol,
                   amd_comgr_symbolizer_info_t *SymbolizeInfo) {
  std::unique_ptr<llvm::MemoryBuffer> Buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(CodeObjectP->Data, CodeObjectP->Size));

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
  llvm::symbolize::Request Request{"", Address};
  auto Printer = std::make_unique<llvm::symbolize::LLVMPrinter>(OS, OS, Config);

  if (IsCode) {
    auto ResOrErr = SymbolizerImpl.symbolizeInlinedCode(
        *CodeObject, {Address, llvm::object::SectionedAddress::UndefSection});
    Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DIInliningInfo());
  } else { // data
    auto ResOrErr = SymbolizerImpl.symbolizeData(
        *CodeObject, {Address, llvm::object::SectionedAddress::UndefSection});
    Printer->print(Request, ResOrErr ? ResOrErr.get() : llvm::DIGlobal());
  }

  PrintSymbol(Result.c_str(), UserData);
  return AMD_COMGR_STATUS_SUCCESS;
}
