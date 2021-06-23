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

#ifndef COMGR_SYMBOLIZER_H
#define COMGR_SYMBOLIZER_H

#include "comgr.h"
#include "llvm/DebugInfo/Symbolize/DIPrinter.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Object/ELFObjectFile.h"
#include <memory>

using namespace llvm::symbolize;
using namespace llvm::object;

namespace COMGR {

typedef void (*PrintSymbolCallback)(const char *, void *);

struct Symbolizer {
  Symbolizer(std::unique_ptr<ObjectFile> &&CodeObject,
             PrintSymbolCallback PrintSymbol)
      : CodeObject(std::move(CodeObject)), PrintSymbol(PrintSymbol) {}

  static amd_comgr_symbolizer_info_t convert(Symbolizer *SymbolizerObj) {
    amd_comgr_symbolizer_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(SymbolizerObj))};
    return Handle;
  }

  static const amd_comgr_symbolizer_info_t
  convert(const Symbolizer *SymbolizerObj) {
    const amd_comgr_symbolizer_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(SymbolizerObj))};
    return Handle;
  }

  static Symbolizer *convert(amd_comgr_symbolizer_info_t SymbolizerInfo) {
    return reinterpret_cast<Symbolizer *>(SymbolizerInfo.handle);
  }

  static amd_comgr_status_t create(DataObject *CodeObjectP,
                                   PrintSymbolCallback PrintSymbol,
                                   amd_comgr_symbolizer_info_t *SymbolizeInfo);

  amd_comgr_status_t symbolize(uint64_t Address, bool IsCode, void *UserData);

private:
  // llvm symbolizer with default options
  LLVMSymbolizer SymbolizerImpl;
  std::unique_ptr<ObjectFile> CodeObject;
  PrintSymbolCallback PrintSymbol;
};
} // namespace COMGR
#endif
