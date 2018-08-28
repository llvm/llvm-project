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
*******************************************************************************/

#include "amd_comgr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

amd_comgr_status_t print_entry(amd_comgr_metadata_node_t key,
                               amd_comgr_metadata_node_t value, void *data) {
  amd_comgr_metadata_kind_t kind;
  amd_comgr_metadata_node_t son;
  amd_comgr_status_t status;
  size_t size;
  char keybuf[50];
  char buf[50];
  int *indent = (int *)data;

  // assume key to be string in this test function
  status = amd_comgr_get_metadata_kind(key, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  if (kind != AMD_COMGR_METADATA_KIND_STRING)
    return AMD_COMGR_STATUS_ERROR;
  status = amd_comgr_get_metadata_string(key, &size, NULL);
  checkError(status, "amd_comgr_get_metadata_string");
  status = amd_comgr_get_metadata_string(key, &size, keybuf);
  checkError(status, "amd_comgr_get_metadata_string");

  status = amd_comgr_get_metadata_kind(value, &kind);
  checkError(status, "amd_comgr_get_metadata_kind");
  for (int i = 0; i < *indent; i++)
    printf("  ");

  switch (kind) {
  case AMD_COMGR_METADATA_KIND_STRING: {
    printf("%s  :  ", size ? keybuf : "");
    status = amd_comgr_get_metadata_string(value, &size, NULL);
    checkError(status, "amd_comgr_get_metadata_string");
    status = amd_comgr_get_metadata_string(value, &size, buf);
    checkError(status, "amd_comgr_get_metadata_string");
    printf(" %s\n", buf);
    break;
  }
  case AMD_COMGR_METADATA_KIND_LIST: {
    *indent += 1;
    status = amd_comgr_get_metadata_list_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_list_size");
    printf("LIST %s %ld entries = \n", keybuf, size);
    for (int i = 0; i < size; i++) {
      status = amd_comgr_index_list_metadata(value, i, &son);
      checkError(status, "amd_comgr_index_list_metadata");
      status = print_entry(key, son, data);
      checkError(status, "print_entry_list");
    }
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  case AMD_COMGR_METADATA_KIND_MAP: {
    *indent += 1;
    status = amd_comgr_get_metadata_map_size(value, &size);
    checkError(status, "amd_comgr_get_metadata_map_size");
    printf("MAP %ld entries = \n", size);
    status = amd_comgr_iterate_map_metadata(value, print_entry, data);
    checkError(status, "amd_comgr_iterate_map_metadata");
    *indent = *indent > 0 ? *indent - 1 : 0;
    break;
  }
  default:
    return AMD_COMGR_STATUS_ERROR;
  } // switch

  return AMD_COMGR_STATUS_SUCCESS;
}

int main(int argc, char *argv[]) {
  char *arg = NULL;
  long size1;
  char *buf;
  amd_comgr_data_t dataIn;
  amd_comgr_status_t status;
  amd_comgr_metadata_kind_t mkind = AMD_COMGR_METADATA_KIND_NULL;

  // Read input file
  size1 = setBuf(TEST_OBJ_DIR "/shared.so", &buf);

  // Create data object
  {
    printf("Test create input data object\n");

    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn);
    checkError(status, "amd_comgr_create_data");

    status = amd_comgr_set_data(dataIn, size1, buf);
    checkError(status, "amd_comgr_set_data");

    status = amd_comgr_set_data_name(dataIn, arg);
    checkError(status, "amd_comgr_set_data_name");
  }

  // Get metadata from data object
  {
    printf("Get metadata from shared.so\n");

    amd_comgr_metadata_node_t meta;
    status = amd_comgr_get_data_metadata(dataIn, &meta);
    checkError(status, "amd_comgr_get_data_metadata");

    // the root must be map
    status = amd_comgr_get_metadata_kind(meta, &mkind);
    checkError(status, "amd_comgr_get_metadata_kind");
    if (mkind != AMD_COMGR_METADATA_KIND_MAP) {
      printf("Root is not map\n");
      exit(1);
    }

    amd_comgr_metadata_node_t metaLookup;
    amd_comgr_metadata_kind_t mkindLookup;
    status = amd_comgr_metadata_lookup(meta, "Version", &metaLookup);
    checkError(status, "amd_comgr_metadata_lookup");
    status = amd_comgr_get_metadata_kind(metaLookup, &mkindLookup);
    checkError(status, "amd_comgr_get_metadata_kind");
    if (mkindLookup != AMD_COMGR_METADATA_KIND_LIST) {
      printf("Lookup of Version should return a list\n");
      exit(1);
    }
    status = amd_comgr_destroy_metadata(metaLookup);
    checkError(status, "amd_comgr_destroy_metadata");

    // print code object metadata
    int indent = 0;
    status = amd_comgr_iterate_map_metadata(meta, print_entry, (void *)&indent);
    checkError(status, "amd_comgr_iterate_map_metadata");
  }

  {
    printf("Cleanup ...\n");
    status = amd_comgr_release_data(dataIn);
    checkError(status, "amd_comgr_release_data");
    free(buf);
  }

  return 0;
}
