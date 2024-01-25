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
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec, DataSetReloc2, DataSetExec2;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;

  SizeSource = setBuf(TEST_OBJ_DIR "/source1.hip", &BufSource);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "source1.hip");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, DataAction,
      DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status =
      amd_comgr_action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  // Get bitcode mangled names
  amd_comgr_data_t DataBc;

  Status = amd_comgr_action_data_get_data(DataSetBc, AMD_COMGR_DATA_KIND_BC, 0,
                                          &DataBc);
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

  const char *bcNames[] = {"_Z7source1Pi"};
  size_t bcNumNames = 1;
  bool bcFound[1] = {false};

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *mName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataBc, I, &Size, mName);
    checkError(Status, "amd_comgr_get_mangled_name");

    for (size_t J = 0; J < bcNumNames; ++J) {
      if (!strcmp(mName, bcNames[J])) {
        bcFound[J] = true;
      }
    }

    free(mName);
  }

  for (size_t I = 0; I < bcNumNames; I++) {
    if (!bcFound[I]) {
      printf("amd_get_mangled_name from bc Failed: "
             "(expected '%s')\n",
             bcNames[I]);
      exit(1);
    }
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

  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);

  Status = amd_comgr_populate_mangled_names(DataExec, &numNames);

  const char *execNames[] = {"_Z7source1Pi", "_Z7source1Pi.kd"};
  size_t execNumNames = 2;
  bool execFound[2] = {false, false};

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *mName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, mName);
    checkError(Status, "amd_comgr_get_mangled_name");

    for (size_t J = 0; J < execNumNames; ++J) {
      if (!strcmp(mName, execNames[J])) {
        execFound[J] = true;
      }
    }

    free(mName);
  }

  for (size_t I = 0; I < execNumNames; I++) {
    if (!execFound[I]) {
      printf("amd_get_mangled_name from bc Failed: "
             "(expected '%s')\n",
             execNames[I]);
      exit(1);
    }
  }

  //
  // Test AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE
  //

  Status = amd_comgr_create_data_set(&DataSetReloc2);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE,
                               DataAction, DataSetIn, DataSetReloc2);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetReloc2,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu source objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_create_data_set(&DataSetExec2);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc2, DataSetExec2);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetExec2,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Get Mangled Names
  amd_comgr_data_t DataExec2;

  Status = amd_comgr_action_data_get_data(
      DataSetExec2, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec2);

  Status = amd_comgr_populate_mangled_names(DataExec2, &numNames);

  for (size_t I = 0; I < execNumNames; ++I) {
    execFound[I] = false;
  }

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, NULL);
    checkError(Status, "amd_comgr_get_mangled_name");

    char *mName = calloc(Size, sizeof(char));
    Status = amd_comgr_get_mangled_name(DataExec, I, &Size, mName);
    checkError(Status, "amd_comgr_get_mangled_name");

    for (size_t J = 0; J < execNumNames; ++J) {
      if (!strcmp(mName, execNames[J])) {
        execFound[J] = true;
      }
    }

    free(mName);
  }

  for (size_t I = 0; I < execNumNames; I++) {
    if (!execFound[I]) {
      printf("amd_get_mangled_name from bc Failed: "
             "(expected '%s')\n",
             execNames[I]);
      exit(1);
    }
  }

  Status = amd_comgr_release_data(DataSource);
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
  Status = amd_comgr_destroy_data_set(DataSetReloc2);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec2);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource);
}
