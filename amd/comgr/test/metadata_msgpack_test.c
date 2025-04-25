//===- metadata_msgpack_test.c --------------------------------------------===//
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
  char *Arg = NULL;
  long Size1;
  char *Buf;
  amd_comgr_data_t DataIn;
  amd_comgr_status_t Status;
  amd_comgr_metadata_kind_t Mkind = AMD_COMGR_METADATA_KIND_NULL;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/shared-v3.so", &Buf);

  // Create data object
  {
    printf("Test create input data object\n");

    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataIn);
    checkError(Status, "amd_comgr_create_data");

    Status = amd_comgr_set_data(DataIn, Size1, Buf);
    checkError(Status, "amd_comgr_set_data");

    Status = amd_comgr_set_data_name(DataIn, Arg);
    checkError(Status, "amd_comgr_set_data_name");
  }

  // Get metadata from data object
  {
    printf("Get metadata from shared.so\n");

    amd_comgr_metadata_node_t Meta;
    Status = amd_comgr_get_data_metadata(DataIn, &Meta);
    checkError(Status, "amd_comgr_get_data_metadata");

    // the root must be map
    Status = amd_comgr_get_metadata_kind(Meta, &Mkind);
    checkError(Status, "amd_comgr_get_metadata_kind");
    if (Mkind != AMD_COMGR_METADATA_KIND_MAP) {
      printf("Root is not map\n");
      exit(1);
    }

    amd_comgr_metadata_node_t MetaLookup;
    amd_comgr_metadata_kind_t MkindLookup;
    Status = amd_comgr_metadata_lookup(Meta, "amdhsa.version", &MetaLookup);
    checkError(Status, "amd_comgr_metadata_lookup");
    Status = amd_comgr_get_metadata_kind(MetaLookup, &MkindLookup);
    checkError(Status, "amd_comgr_get_metadata_kind");
    if (MkindLookup != AMD_COMGR_METADATA_KIND_LIST) {
      printf("Lookup of Version should return a list\n");
      exit(1);
    }
    Status = amd_comgr_destroy_metadata(MetaLookup);
    checkError(Status, "amd_comgr_destroy_metadata");

    // print code object metadata
    int Indent = 0;
    Status = amd_comgr_iterate_map_metadata(Meta, printEntry, (void *)&Indent);
    checkError(Status, "amd_comgr_iterate_map_metadata");

    Status = amd_comgr_destroy_metadata(Meta);
    checkError(Status, "amd_comgr_destroy_metadata");
  }

  {
    printf("Cleanup ...\n");
    Status = amd_comgr_release_data(DataIn);
    checkError(Status, "amd_comgr_release_data");
    free(Buf);
  }

  return 0;
}
