//===- name-expression-map.c - Driver for name expression map lit tests ---===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Loads a code object file as an executable data object and runs
/// amd_comgr_populate_name_expression_map on it, reporting the status. Used by
/// lit tests to feed crafted (including malformed) code objects to the parser.
///
//===----------------------------------------------------------------------===//

#include "common.h"

int main(int argc, char *argv[]) {
  if (argc < 2)
    fail("usage: name-expression-map <code-object-file>");

  char *Buf;
  size_t Size = (size_t)setBuf(argv[1], &Buf);

  amd_comgr_data_t Data;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Data));
  amd_comgr_(set_data(Data, Size, Buf));

  size_t Count = 0;
  amd_comgr_status_t Status =
      amd_comgr_populate_name_expression_map(Data, &Count);

  if (Status == AMD_COMGR_STATUS_SUCCESS)
    printf("RESULT: SUCCESS count=%zu\n", Count);
  else if (Status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)
    printf("RESULT: INVALID_ARGUMENT\n");
  else
    printf("RESULT: ERROR\n");

  amd_comgr_(release_data(Data));
  free(Buf);
  return 0;
}
