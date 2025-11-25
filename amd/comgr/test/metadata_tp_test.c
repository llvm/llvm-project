//===- metadata_tp_test.c -------------------------------------------------===//
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
  amd_comgr_status_t Status;

  // how many isa_names do we support?
  size_t IsaCounts;
  Status = amd_comgr_get_isa_count(&IsaCounts);
  checkError(Status, "amd_comgr_get_isa_count");
  printf("isa count = %zu\n\n", IsaCounts);

  // print the list
  printf("*** List of ISA names supported:\n");
  for (size_t I = 0; I < IsaCounts; I++) {
    const char *Name;
    Status = amd_comgr_get_isa_name(I, &Name);
    checkError(Status, "amd_comgr_get_isa_name");
    printf("%zu: %s\n", I, Name);
    amd_comgr_metadata_node_t Meta;
    Status = amd_comgr_get_isa_metadata(Name, &Meta);
    checkError(Status, "amd_comgr_get_isa_metadata");
    int Indent = 1;
    Status = amd_comgr_iterate_map_metadata(Meta, printEntry, (void *)&Indent);
    checkError(Status, "amd_comgr_iterate_map_metadata");
    Status = amd_comgr_destroy_metadata(Meta);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

  return 0;
}
