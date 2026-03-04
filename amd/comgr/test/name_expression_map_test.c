//===- name_expression_map_test.c -----------------------------------------===//
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
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec, DataSetReloc2, DataSetExec2;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;
  const char *CompileOptions[] = {"-nogpulib", "-nogpuinc"};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

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
  Status =
      amd_comgr_action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(DataAction, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, DataAction,
      DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  // Check name_expression_map for Bitcodes
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

  size_t NumNames;
  Status = amd_comgr_populate_name_expression_map(DataBc, &NumNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (NumNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu bitcode names (expected 2)\n",
           NumNames);
    exit(1);
  }

  const char *NameExpressions[] = {
      "my_kernel_BOO<static_cast<int>(2+1),float >",
      "my_kernel_FOO<static_cast<int>(2+1),float >"};
  const char *SymbolNames[] = {"_Z13my_kernel_BOOILi3EfEvPT0_",
                               "_Z13my_kernel_FOOILi3EfEvPT0_"};

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataBc, &Size, NameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *SymbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataBc, &Size, NameExpressions[I], SymbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (!SymbolNames[I]) {
      printf("Failed, symbolNames[%ld] NULL\n", I);
      return 1;
    }

    if (strcmp(SymbolName, SymbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from bc Failed: "
             "produced '%s' (expected '%s')\n",
             SymbolName, SymbolNames[I]);
      exit(1);
    }

    free(SymbolName);
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

  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);
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

  Status = amd_comgr_populate_name_expression_map(DataExec, &NumNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (NumNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu code object names (expected 2)\n",
           NumNames);
    exit(1);
  }

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec, &Size, NameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *SymbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec, &Size, NameExpressions[I], SymbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (!SymbolNames[I]) {
      printf("Failed, symbolNames[%ld] NULL\n", I);
      return 1;
    }

    if (strcmp(SymbolName, SymbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from exec Failed: "
             "produced '%s' (expected '%s')\n",
             SymbolName, SymbolNames[I]);
      exit(1);
    }

    free(SymbolName);
  }

  //
  // Test AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE
  //
  Status = amd_comgr_create_data_set(&DataSetReloc2);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_option_list(DataAction, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");
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

  Status = amd_comgr_populate_name_expression_map(DataExec2, &NumNames);
  checkError(Status, "amd_comgr_populate_name_expression_map");

  if (NumNames != 2) {
    printf("amd_populate_name_expression_map Failed: "
           "produced %zu code object names (expected 2)\n",
           NumNames);
    exit(1);
  }

  for (size_t I = 0; I < NumNames; ++I) {
    size_t Size;
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec2, &Size, NameExpressions[I], NULL);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    char *SymbolName = calloc(Size, sizeof(char));
    Status = amd_comgr_map_name_expression_to_symbol_name(
        DataExec2, &Size, NameExpressions[I], SymbolName);
    checkError(Status, "amd_map_name_expression_to_symbol_name");

    if (!SymbolNames[I]) {
      printf("Failed, symbolNames[%ld] NULL\n", I);
      return 1;
    }

    if (strcmp(SymbolName, SymbolNames[I])) {
      printf("amd_comgr_map_name_expression_to_symbol_name from exec Failed: "
             "produced '%s' (expected '%s')\n",
             SymbolName, SymbolNames[I]);
      exit(1);
    }

    free(SymbolName);
  }

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataBc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataReloc2);
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
