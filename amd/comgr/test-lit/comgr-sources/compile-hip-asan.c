//===- compile-hip-asan.c -------------------------------------------------===//
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
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetReloc, DataSetExec;
  amd_comgr_action_info_t DataAction;
  size_t Count;
  const char *CompileOptions[] = {"-nogpuinc", "-fsanitize=address"};
  size_t CompileOptionsCount = sizeof(CompileOptions) / sizeof(CompileOptions[0]);

  if (argc != 3) {
    fprintf(stderr, "Usage: compile-hip-asan <input.hip> <output.bin>\n");
    exit(1);
  }

  SizeSource = setBuf(argv[1], &BufSource);

  amd_comgr_(create_data_set(&DataSetIn));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, SizeSource, BufSource));
  amd_comgr_(set_data_name(DataSource, "source1.hip"));
  amd_comgr_(data_set_add(DataSetIn, DataSource));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP));
  amd_comgr_(action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900"));
  amd_comgr_(action_info_set_option_list(DataAction, CompileOptions,
                                         CompileOptionsCount));
  amd_comgr_(action_info_set_device_lib_linking(DataAction, true));

  amd_comgr_(create_data_set(&DataSetBc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                       DataAction, DataSetIn, DataSetBc));
  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_(create_data_set(&DataSetReloc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, DataAction,
                       DataSetBc, DataSetReloc));
  amd_comgr_(
      action_data_count(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count));
  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu relocatable objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_(create_data_set(&DataSetExec));
  // Keep -fsanitize=address for linking so the ASAN runtime gets linked
  amd_comgr_(do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                       DataAction, DataSetReloc, DataSetExec));

  amd_comgr_(
      action_data_count(DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, &Count));
  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_data_t DataExec;
  amd_comgr_(action_data_get_data(DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE,
                                  0, &DataExec));
  dumpData(DataExec, argv[2]);

  amd_comgr_(release_data(DataSource));
  amd_comgr_(release_data(DataExec));
  amd_comgr_(destroy_data_set(DataSetIn));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_data_set(DataSetReloc));
  amd_comgr_(destroy_data_set(DataSetExec));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSource);
  return 0;
}
