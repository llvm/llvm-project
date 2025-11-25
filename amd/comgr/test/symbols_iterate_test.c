//===- symbols_iterate_test.c ---------------------------------------------===//
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

int main(int argc, char *argv[]) {
  long Size;
  char *Buf;
  amd_comgr_data_t DataObject;
  amd_comgr_status_t Status;
  int Count = 1;

  Size = setBuf(TEST_OBJ_DIR "/shared.so", &Buf);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataObject);
  checkError(Status, "amd_comgr_create_data");

  Status = amd_comgr_set_data(DataObject, Size, Buf);
  checkError(Status, "amd_comgr_set_data");

  Status = amd_comgr_iterate_symbols(DataObject, printSymbol, &Count);
  checkError(Status, "amd_comgr_iterate_symbols");

  Status = amd_comgr_release_data(DataObject);
  checkError(Status, "amd_comgr_release_data");
  free(Buf);

  return 0;
}
