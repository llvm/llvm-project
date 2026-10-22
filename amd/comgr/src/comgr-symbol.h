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
#include "llvm/ADT/StringMap.h"
#include "llvm/Object/ObjectFile.h"

namespace COMGR {

struct DataObject;

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

// Everything amd_comgr_symbol_get_info reports, so a name lookup never has to
// revisit the object file.
struct SymbolInfo {
  uint64_t Value;
  uint64_t Size;
  amd_comgr_symbol_type_t Type;
  bool Undefined;
};

class SymbolHelper {

public:
  amd_comgr_symbol_type_t mapToComgrSymbolType(uint8_t ELFSymbolType);

  // Both return nullptr when DataP->Data is not an object file.
  llvm::object::ObjectFile *getCachedObjectFile(DataObject *DataP);
  const llvm::StringMap<SymbolInfo> *getSymbolIndex(DataObject *DataP);

  SymbolContext *createBinary(DataObject *DataP, const char *Name);

  amd_comgr_status_t
  iterateTable(DataObject *DataP,
               amd_comgr_status_t (*Callback)(amd_comgr_symbol_t, void *),
               void *UserData);

}; // SymbolHelper

} // namespace COMGR

#endif
