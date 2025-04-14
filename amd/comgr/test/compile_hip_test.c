//===- compile_hip_test.c -------------------------------------------------===//
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

int main(int Argc, char *Argv[]) {
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSrc;
  amd_comgr_data_set_t DataSetSrc, DataSetBc, DataSetLinkedBc, DataSetAsm,
      DataSetReloc, DataSetExec;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;
  const char *CompileOptions[] = {"-nogpulib", "-nogpuinc"};
  size_t CompileOptionsCount =
      sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  SizeSource = setBuf(TEST_OBJ_DIR "/source1.hip", &BufSource);

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSrc);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSrc, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSrc, "source1.hip");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_create_data_set(&DataSetSrc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_data_set_add(DataSetSrc, DataSrc);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(ActionInfo,
                                              "amdgcn-amd-amdhsa--gfx906");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_option_list(ActionInfo, CompileOptions,
                                                 CompileOptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(
      AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, ActionInfo,
      DataSetSrc, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetLinkedBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfo,
                               DataSetBc, DataSetLinkedBc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetAsm);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               ActionInfo, DataSetLinkedBc, DataSetAsm);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               ActionInfo, DataSetLinkedBc, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               ActionInfo, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_destroy_data_set(DataSetSrc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinkedBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetAsm);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_release_data(DataSrc);
  checkError(Status, "amd_comgr_release_data");

  free(BufSource);
}
