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
  char *BufSource1, *BufSource2, *BufInclude;
  size_t SizeSource1, SizeSource2, SizeInclude;
  amd_comgr_data_t DataSource1, DataSource2, DataInclude;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;
  const char *CodeGenOptions[] = {"-mllvm", "-amdgpu-early-inline-all"};
  size_t CodeGenOptionsCount =
      sizeof(CodeGenOptions) / sizeof(CodeGenOptions[0]);

  SizeSource1 = setBuf(TEST_OBJ_DIR "/source1.cl", &BufSource1);
  SizeSource2 = setBuf(TEST_OBJ_DIR "/source2.cl", &BufSource2);
  SizeInclude = setBuf(TEST_OBJ_DIR "/include-a.h", &BufInclude);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource1, SizeSource1, BufSource1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource1, "source1.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource1);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource2);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource2, SizeSource2, BufSource2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource2, "source2.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource2);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude, SizeInclude, BufInclude);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude, "include-a.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx803");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(DataAction, CodeGenOptions,
                                                 CodeGenOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status =
      amd_comgr_action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 2) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 2)\n",
           Count);
    exit(1);
  }

  // Get bitcode mangled names
  amd_comgr_data_t DataBc;

  Status = amd_comgr_action_data_get_data(DataSetBc,
                                          AMD_COMGR_DATA_KIND_BC,
                                          0, &DataBc);
  checkError(Status, "amd_comgr_action_data_get_data");

#if 0
  // write bitcode
  {
    size_t bytes_size = 0;
    char *bytes = NULL;

    Status = amd_comgr_get_data(DataBc, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    bytes = (char *) malloc(bytes_size);

    Status = amd_comgr_get_data(DataBc, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    const char *bitcode_file = "comgr_mangled.bc";
    FILE *file = fopen(bitcode_file, "wb");

    if (file)
      fwrite(bytes, bytes_size, 1, file);
    else
      return AMD_COMGR_STATUS_ERROR;

    fclose(file);
    free(bytes);
  }
#endif

  size_t numNames;
  Status = amd_comgr_populate_mangled_names(DataBc, &numNames);
  checkError(Status, "amd_comgr_populate_mangled_names");

  if (numNames != 2) {
    printf("amd_populate_mangled_names Failed: "
           "produced %zu bitcode names (expected 2)\n",
           Count);
    exit(1);
  }

  const char *bcNames[] = {"source1", "source2"};

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *mName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, mName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (strcmp(mName, bcNames[I])) {
      printf("amd_get_mangled_name from bc Failed: "
             "produced '%s' (expected '%s')\n",
             mName, bcNames[I]);
      exit(1);
    }

    free(mName);
  }

  Status = amd_comgr_create_data_set(&DataSetLinked);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, DataAction,
                               DataSetBc, DataSetLinked);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetLinked, AMD_COMGR_DATA_KIND_BC,
                                       &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_BC_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               DataAction, DataSetLinked, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu source objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Get Mangled Names
  amd_comgr_data_t DataExec;

  Status = amd_comgr_action_data_get_data(DataSetExec,
                                          AMD_COMGR_DATA_KIND_EXECUTABLE,
                                          0, &DataExec);

  Status = amd_comgr_populate_mangled_names(DataExec, &numNames);

  if (numNames != 4) {
    printf("amd_populate_mangled_names Failed: "
           "produced %zu executable names (expected 4)\n",
           numNames);
    exit(1);
  }

  const char *execNames[] = {"source1", "source1.kd", "source2", "source2.kd"};

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *mName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, mName);
    checkError(Status, "amd_comgr_get_mangled_name");

    if (strcmp(mName, execNames[I])) {
      printf("amd_get_mangled_name from executable Failed: "
             "produced '%s' (expected '%s')\n",
             mName, execNames[I]);
      exit(1);
    }

    free(mName);
  }

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataBc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource1);
  free(BufSource2);
  free(BufInclude);
}
