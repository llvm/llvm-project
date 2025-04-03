#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *BufSpv;
  size_t SizeSpv;
  amd_comgr_data_t DataSpv;
  amd_comgr_data_set_t DataSetSpv, DataSetReloc;
  amd_comgr_action_info_t DataAction;
  size_t Count;

  if (argc != 3) {
    fprintf(stderr, "Usage: spirv-to-reloc file.spv file.o\n");
    exit(1);
  }

  SizeSpv = setBuf(argv[1], &BufSpv);

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SPIRV, &DataSpv));
  amd_comgr_(set_data(DataSpv, SizeSpv, BufSpv));
  amd_comgr_(set_data_name(DataSpv, "file.spv"));

  amd_comgr_(create_data_set(&DataSetSpv));
  amd_comgr_(data_set_add(DataSetSpv, DataSpv));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP));
  amd_comgr_(action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900"));

  amd_comgr_(create_data_set(&DataSetReloc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE,
                       DataAction, DataSetSpv, DataSetReloc));

  amd_comgr_(
      action_data_count(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE Failed: "
           "produced %zu RELOCATABLE objects (expected 1)\n",
           Count);
    exit(1);
  }

  amd_comgr_data_t DataReloc;
  amd_comgr_(action_data_get_data(DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE,
                                  0, &DataReloc));
  dumpData(DataReloc, argv[2]);

  amd_comgr_(release_data(DataSpv));
  amd_comgr_(destroy_data_set(DataSetSpv));
  amd_comgr_(destroy_data_set(DataSetReloc));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSpv);
}
