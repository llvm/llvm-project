//===- compile_source_to_executable.c -------------------------------------===//
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

  // OpenCL
  {
    char *BufSource;
    size_t SizeSource;
    amd_comgr_data_t DataSource;
    amd_comgr_data_set_t DataSetIn, DataSetExec;
    amd_comgr_action_info_t DataAction;
    amd_comgr_status_t Status;

    // Create OpenCL source data set
    SizeSource = setBuf(TEST_OBJ_DIR "/source1.cl", &BufSource);

    Status = amd_comgr_create_data_set(&DataSetIn);
    checkError(Status, "amd_comgr_create_data_set");

    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataSource, "source1.cl");
    checkError(Status, "amd_comgr_set_data_name");
    Status = amd_comgr_data_set_add(DataSetIn, DataSource);
    checkError(Status, "amd_comgr_data_set_add");

    // Set up ActionInfo
    Status = amd_comgr_create_action_info(&DataAction);
    checkError(Status, "amd_comgr_create_action_info");
    Status = amd_comgr_action_info_set_language(DataAction,
                                                AMD_COMGR_LANGUAGE_OPENCL_1_2);
    checkError(Status, "amd_comgr_action_info_set_language");
    Status = amd_comgr_action_info_set_isa_name(DataAction,
                                                "amdgcn-amd-amdhsa--gfx900");
    checkError(Status, "amd_comgr_action_info_set_isa_name");

    // Compile source to executable
    Status = amd_comgr_create_data_set(&DataSetExec);
    checkError(Status, "amd_comgr_create_data_set");

    Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
    checkError(Status, "amd_comgr_action_info_set_option_list");

    Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE,
                                 DataAction, DataSetIn, DataSetExec);
    checkError(Status, "amd_comgr_do_action");

    size_t Count;
    Status = amd_comgr_action_data_count(
        DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 1) {
      printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE Failed: "
             "produced %zu executable objects from source (expected 1)\n",
             Count);
      exit(1);
    }

    Status = amd_comgr_release_data(DataSource);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_destroy_data_set(DataSetIn);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_data_set(DataSetExec);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_action_info(DataAction);
    checkError(Status, "amd_comgr_destroy_action_info");
    free(BufSource);
  }

  // Re-enable post https://github.com/llvm/llvm-project/pull/85672
#if 0
  // HIP
  {
    char *BufSource;
    size_t SizeSource;
    amd_comgr_data_t DataSource;
    amd_comgr_data_set_t DataSetIn, DataSetExec;
    amd_comgr_action_info_t DataAction;
    amd_comgr_status_t Status;

    // Create HIP source data set
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

    // Set up ActionInfo
    Status = amd_comgr_create_action_info(&DataAction);
    checkError(Status, "amd_comgr_create_action_info");
    Status = amd_comgr_action_info_set_language(DataAction,
                                                AMD_COMGR_LANGUAGE_HIP);
    checkError(Status, "amd_comgr_action_info_set_language");
    Status = amd_comgr_action_info_set_isa_name(DataAction,
                                                "amdgcn-amd-amdhsa--gfx900");
    checkError(Status, "amd_comgr_action_info_set_isa_name");

    // Compile source to executable
    Status = amd_comgr_create_data_set(&DataSetExec);
    checkError(Status, "amd_comgr_create_data_set");

    Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
    checkError(Status, "amd_comgr_action_info_set_option_list");

    Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE,
                                 DataAction, DataSetIn, DataSetExec);
    checkError(Status, "amd_comgr_do_action");

    size_t Count;
    Status = amd_comgr_action_data_count(DataSetExec,
                                         AMD_COMGR_DATA_KIND_EXECUTABLE,
                                         &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 1) {
      printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE Failed: "
             "produced %zu executable objects from source (expected 1)\n",
             Count);
      exit(1);
    }

    Status = amd_comgr_release_data(DataSource);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_destroy_data_set(DataSetIn);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_data_set(DataSetExec);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_action_info(DataAction);
    checkError(Status, "amd_comgr_destroy_action_info");
    free(BufSource);
  }
#endif

  // Bitcode
  {
    char *BufSource;
    size_t SizeSource;
    amd_comgr_data_t DataSource;
    amd_comgr_data_set_t DataSetIn, DataSetExec;
    amd_comgr_action_info_t DataAction;
    amd_comgr_status_t Status;

    // Create Bitcode source data set
    SizeSource = setBuf(TEST_OBJ_DIR "/source1.bc", &BufSource);

    Status = amd_comgr_create_data_set(&DataSetIn);
    checkError(Status, "amd_comgr_create_data_set");

    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &DataSource);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataSource, "source1.bc");
    checkError(Status, "amd_comgr_set_data_name");
    Status = amd_comgr_data_set_add(DataSetIn, DataSource);
    checkError(Status, "amd_comgr_data_set_add");

    // Set up ActionInfo
    Status = amd_comgr_create_action_info(&DataAction);
    checkError(Status, "amd_comgr_create_action_info");
    Status = amd_comgr_action_info_set_language(DataAction,
                                                AMD_COMGR_LANGUAGE_LLVM_IR);
    checkError(Status, "amd_comgr_action_info_set_language");
    Status = amd_comgr_action_info_set_isa_name(DataAction,
                                                "amdgcn-amd-amdhsa--gfx900");
    checkError(Status, "amd_comgr_action_info_set_isa_name");

    // Compile source to executable
    Status = amd_comgr_create_data_set(&DataSetExec);
    checkError(Status, "amd_comgr_create_data_set");

    Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
    checkError(Status, "amd_comgr_action_info_set_option_list");

    Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE,
                                 DataAction, DataSetIn, DataSetExec);
    checkError(Status, "amd_comgr_do_action");

    size_t Count;
    Status = amd_comgr_action_data_count(
        DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");

    if (Count != 1) {
      printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE Failed: "
             "produced %zu executable objects from bitcode (expected 1)\n",
             Count);
      exit(1);
    }

    Status = amd_comgr_release_data(DataSource);
    checkError(Status, "amd_comgr_release_data");
    Status = amd_comgr_destroy_data_set(DataSetIn);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_data_set(DataSetExec);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_action_info(DataAction);
    checkError(Status, "amd_comgr_destroy_action_info");
    free(BufSource);
  } // end Bitcode
}
