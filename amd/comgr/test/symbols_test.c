//===- symbols_test.c -----------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void expectSymbol(const char *ObjectFilename, const char *SymbolName,
                  amd_comgr_symbol_type_t ExpectedType) {
  long Size;
  char *Buf;
  amd_comgr_data_t DataObject;
  amd_comgr_symbol_t Symbol;
  amd_comgr_status_t Status;

  Size = setBuf(ObjectFilename, &Buf);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataObject);
  checkError(Status, "amd_comgr_create_data");

  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_symbol_lookup(DataObject, SymbolName, &Symbol);
  checkError(Status, "amd_comgr_symbol_lookup");

  amd_comgr_symbol_type_t Type;
  Status = amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_TYPE,
                                     (void *)&Type);
  checkError(Status, "amd_comgr_symbol_get_info");

  if (Type != ExpectedType) {
    fail("unexpected symbol type for symbol %s: expected %d, saw %d\n",
         SymbolName, ExpectedType, Type);
  }

  Status = amd_comgr_release_data(DataObject);
  checkError(Status, "amd_comgr_release_data");
  free(Buf);
}

int main(int argc, char *argv[]) {
  expectSymbol(TEST_OBJ_DIR "/shared-v2.so",
               "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
               AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL);
  expectSymbol(TEST_OBJ_DIR "/shared-v3.so",
               "bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
               AMD_COMGR_SYMBOL_TYPE_FUNC);
  expectSymbol(TEST_OBJ_DIR "/shared-v2.so", "foo",
               AMD_COMGR_SYMBOL_TYPE_OBJECT);
  expectSymbol(TEST_OBJ_DIR "/shared-v3.so", "foo",
               AMD_COMGR_SYMBOL_TYPE_OBJECT);
  return 0;
}
