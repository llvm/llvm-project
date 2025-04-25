//===- comgr-symbolizer.h - Symbolizer implementation ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
             PrintSymbolCallback PrintSymbol);
  ~Symbolizer();

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
  std::unique_ptr<LLVMSymbolizer> SymbolizerImpl;
  std::unique_ptr<ObjectFile> CodeObject;
  PrintSymbolCallback PrintSymbol;
};
} // namespace COMGR
#endif
