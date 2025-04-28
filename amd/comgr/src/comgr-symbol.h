//===- comgr-symbol.h - Symbol lookup -------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_SYMBOL_H_
#define COMGR_SYMBOL_H_

#include "amd_comgr.h"
#include "llvm/Object/ObjectFile.h"

namespace COMGR {

struct SymbolContext {
  SymbolContext();
  ~SymbolContext();

  amd_comgr_status_t setName(llvm::StringRef Name);

  char *Name;
  amd_comgr_symbol_type_t Type;
  uint64_t Size;
  bool Undefined;
  uint64_t Value;
};

class SymbolHelper {

public:
  amd_comgr_symbol_type_t mapToComgrSymbolType(uint8_t ELFSymbolType);

  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
  createBinary(llvm::StringRef InBuffer);

  SymbolContext *createBinary(llvm::StringRef InBuffer, const char *Name,
                              amd_comgr_data_kind_t Kind);

  amd_comgr_status_t
  iterateTable(llvm::StringRef InBuffer, amd_comgr_data_kind_t Kind,
               amd_comgr_status_t (*Callback)(amd_comgr_symbol_t, void *),
               void *UserData);

}; // SymbolHelper

} // namespace COMGR

#endif
