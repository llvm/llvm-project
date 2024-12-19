#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetPch, DataSetBc;
  amd_comgr_action_info_t DataAction;
  const char *CodeGenOptions[] = {
    "-mcode-object-version=5", "-mllvm", "-amdgpu-prelink"};
  size_t CodeGenOptionsCount =
      sizeof(CodeGenOptions) / sizeof(CodeGenOptions[0]);
  if (argc != 4) {
    fprintf(stderr, "Usage: source-to-bc-with-device-libs file.cl -o file.bc\n");
    exit(1);
  }

  SizeSource = setBuf(argv[1], &BufSource);

  amd_comgr_(create_data_set(&DataSetIn));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, SizeSource, BufSource));
  amd_comgr_(set_data_name(DataSource, "device-lib-linking.cl"));
  amd_comgr_(data_set_add(DataSetIn, DataSource));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction,
                                      AMD_COMGR_LANGUAGE_OPENCL_1_2));
  amd_comgr_(action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900"));
  amd_comgr_(create_data_set(&DataSetPch));

  amd_comgr_(do_action(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS, DataAction,
                               DataSetIn, DataSetPch));

  size_t Count;
  amd_comgr_(action_data_count(DataSetPch,
                               AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS Failed: "
           "produced %zu precompiled header objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_(create_data_set(&DataSetBc));
  amd_comgr_(action_info_set_option_list(DataAction, CodeGenOptions,
                                                 CodeGenOptionsCount));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                       DataAction, DataSetPch, DataSetBc));

  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_data_t DataBc;
  amd_comgr_(action_data_get_data(DataSetBc, AMD_COMGR_DATA_KIND_BC, 0,
                                  &DataBc));
  dumpData(DataBc, argv[3]);

  amd_comgr_(release_data(DataSource));
  amd_comgr_(release_data(DataBc));
  amd_comgr_(destroy_data_set(DataSetIn));
  amd_comgr_(destroy_data_set(DataSetPch));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSource);
}
