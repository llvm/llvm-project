//===- compile_minimal_test.c ---------------------------------------------===//
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
  char *BufSource1, *BufSource2, *BufInclude;
  size_t SizeSource1, SizeSource2, SizeInclude;
  amd_comgr_data_t DataSource1, DataSource2, DataInclude;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;
  const char *CodeGenOptions[] = {"-mllvm", "--color"};
  size_t CodeGenOptionsCount =
      sizeof(CodeGenOptions) / sizeof(CodeGenOptions[0]);

  SizeSource1 = setBuf(TEST_OBJ_DIR "/source1.cl", &BufSource1);
  SizeSource2 = setBuf(TEST_OBJ_DIR "/source2.cl", &BufSource2);
  SizeInclude = setBuf(TEST_OBJ_DIR "/include-macro.h", &BufInclude);

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
  Status = amd_comgr_set_data_name(DataInclude, "include-macro.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
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

  Status = amd_comgr_action_info_set_device_lib_linking(DataAction, true);
  checkError(Status, "amd_comgr_action_info_set_device_lib_linking");

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

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude);
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
