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

  SizeSource = setBuf(TEST_OBJ_DIR "/name-expression.hip", &BufSource);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "name-expression.hip");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  // Check name_expression_map for Bitcodes
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

    const char *bitcode_file = "comgr_name_expression.bc";
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
  Status = amd_comgr_populate_name_expression_map(DataBc, &numNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (numNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu bitcode names (expected 2)\n",
           numNames);
    exit(1);
  }

  char *nameExpressions[] = {"my_kernel_BOO<static_cast<int>(2+1),float >",
                             "my_kernel_FOO<static_cast<int>(2+1),float >"};
  char *symbolNames[] = {"_Z13my_kernel_BOOILi3EfEvPT0_",
                         "_Z13my_kernel_FOOILi3EfEvPT0_"};

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
      DataBc, &Size, nameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *symbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
      DataBc, &Size, nameExpressions[I], symbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (strcmp(symbolName, symbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from bc Failed: "
             "produced '%s' (expected '%s')\n",
             symbolName, symbolNames[I]);
      exit(1);
    }

    free(symbolName);
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


  // Check name_expression_map for Code Objects
  amd_comgr_data_t DataExec;

  Status = amd_comgr_action_data_get_data(DataSetExec,
                                          AMD_COMGR_DATA_KIND_EXECUTABLE,
                                          0, &DataExec);
#if 0
  // write code object
  {
    size_t bytes_size = 0;
    char *bytes = NULL;

    Status = amd_comgr_get_data(DataExec, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    bytes = (char *) malloc(bytes_size);

    Status = amd_comgr_get_data(DataExec, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    const char *code_object_file = "comgr_name_expression.o";
    FILE *file = fopen(code_object_file, "wb");

    if (file)
      fwrite(bytes, bytes_size, 1, file);
    else
      return AMD_COMGR_STATUS_ERROR;

    fclose(file);
    free(bytes);
  }
#endif

  Status = amd_comgr_populate_name_expression_map(DataExec, &numNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (numNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu code object names (expected 2)\n",
           numNames);
    exit(1);
  }

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec, &Size, nameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *symbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec, &Size, nameExpressions[I], symbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (strcmp(symbolName, symbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from exec Failed: "
             "produced '%s' (expected '%s')\n",
             symbolName, symbolNames[I]);
      exit(1);
    }

    free(symbolName);
  }

  //
  // Test AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE
  //
  Status = amd_comgr_create_data_set(&DataSetReloc2);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE,
                               DataAction, DataSetIn, DataSetReloc2);
  checkError(Status, "amd_comgr_do_action");

  // Check name_expression_map for Bitcodes
  amd_comgr_data_t DataReloc2;

  Status = amd_comgr_action_data_get_data(
      DataSetReloc2, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &DataReloc2);
  checkError(Status, "amd_comgr_action_data_get_data");

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

  // Check name_expression_map for Code Objects
  amd_comgr_data_t DataExec2;

  Status = amd_comgr_action_data_get_data(
      DataSetExec2, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec2);
#if 0
  // write code object
  {
    size_t bytes_size = 0;
    char *bytes = NULL;

    Status = amd_comgr_get_data(DataExec2, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    bytes = (char *) malloc(bytes_size);

    Status = amd_comgr_get_data(DataExec2, &bytes_size, bytes);
    checkError(Status, "amd_comgr_get_data");

    const char *code_object_file = "comgr_name_expression.o";
    FILE *file = fopen(code_object_file, "wb");

    if (file)
      fwrite(bytes, bytes_size, 1, file);
    else
      return AMD_COMGR_STATUS_ERROR;

    fclose(file);
    free(bytes);
  }
#endif

  Status = amd_comgr_populate_name_expression_map(DataExec2, &numNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (numNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu code object names (expected 2)\n",
           numNames);
    exit(1);
  }

  for (size_t I = 0; I < numNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec2, &Size, nameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *symbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec2, &Size, nameExpressions[I], symbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (strcmp(symbolName, symbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from exec Failed: "
             "produced '%s' (expected '%s')\n",
             symbolName, symbolNames[I]);
      exit(1);
    }

    free(symbolName);
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
