#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tests the AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC action
//     Accepts one or more .spv files, and returns one or more .bc files

int main(int argc, char *argv[]) {
  char *BufSpirv;
  size_t SizeSpirv;
  amd_comgr_data_t DataSpirv;
  amd_comgr_data_set_t DataSetSpirv, DataSetBc;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;

  if (argc != 4) {
    fprintf(stderr, "Usage: spirv-translator file.spv -o file.spv.bc\n");
    exit(1);
  }

  SizeSpirv = setBuf(argv[1], &BufSpirv);

  amd_comgr_(create_data_set(&DataSetSpirv));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SPIRV, &DataSpirv));
  amd_comgr_(set_data(DataSpirv, SizeSpirv, BufSpirv));
  amd_comgr_(set_data_name(DataSpirv, "source.spv"));
  amd_comgr_(data_set_add(DataSetSpirv, DataSpirv));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(create_data_set(&DataSetBc));

  amd_comgr_(do_action(AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC,
                      DataAction, DataSetSpirv, DataSetBc));

  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Write bitcode to file
  amd_comgr_data_t DataSpirvBc;

  amd_comgr_(action_data_get_data(
      DataSetBc, AMD_COMGR_DATA_KIND_BC, 0, &DataSpirvBc));

  dumpData(DataSpirvBc, argv[3]);

  amd_comgr_(release_data(DataSpirv));
  amd_comgr_(destroy_data_set(DataSetSpirv));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSpirv);
}
