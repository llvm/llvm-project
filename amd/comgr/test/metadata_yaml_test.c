/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

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
  Size1 = setBuf(TEST_OBJ_DIR "/shared-v2.so", &Buf);

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
    printf("Get metadata from shared-v2.so\n");

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
    Status = amd_comgr_metadata_lookup(Meta, "Version", &MetaLookup);
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
