//===- spirv-translator.c -------------------------------------------------===//
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

// Tests the AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC action
//     Accepts one or more .spv files, and returns one or more .bc files
//     Optional: --isa <isa_name> to set the ISA for offload arch forwarding

int main(int argc, char *argv[]) {
  char *BufSpirv;
  size_t SizeSpirv;
  amd_comgr_data_t DataSpirv;
  amd_comgr_data_set_t DataSetSpirv, DataSetBc;
  amd_comgr_action_info_t DataAction;
  size_t Count;

  // Parse arguments: spirv-translator [--isa <name>] file.spv -o file.bc
  const char *IsaName = NULL;
  const char *InputFile = NULL;
  const char *OutputFile = NULL;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--isa") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "--isa requires an argument\n");
        exit(1);
      }
      IsaName = argv[++i];
    } else if (strcmp(argv[i], "-o") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "-o requires an argument\n");
        exit(1);
      }
      OutputFile = argv[++i];
    } else if (!InputFile) {
      InputFile = argv[i];
    } else {
      fprintf(stderr,
              "Usage: spirv-translator [--isa <name>] file.spv -o file.bc\n");
      exit(1);
    }
  }

  if (!InputFile || !OutputFile) {
    fprintf(stderr,
            "Usage: spirv-translator [--isa <name>] file.spv -o file.bc\n");
    exit(1);
  }

  SizeSpirv = setBuf(InputFile, &BufSpirv);

  amd_comgr_(create_data_set(&DataSetSpirv));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SPIRV, &DataSpirv));
  amd_comgr_(set_data(DataSpirv, SizeSpirv, BufSpirv));
  amd_comgr_(set_data_name(DataSpirv, "source.spv"));
  amd_comgr_(data_set_add(DataSetSpirv, DataSpirv));

  amd_comgr_(create_action_info(&DataAction));

  if (IsaName)
    amd_comgr_(action_info_set_isa_name(DataAction, IsaName));

  amd_comgr_(create_data_set(&DataSetBc));

  amd_comgr_(do_action(AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC, DataAction,
                       DataSetSpirv, DataSetBc));

  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_BC, &Count));

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           Count);
    exit(1);
  }

  // Write bitcode to file
  amd_comgr_data_t DataSpirvBc;

  amd_comgr_(
      action_data_get_data(DataSetBc, AMD_COMGR_DATA_KIND_BC, 0, &DataSpirvBc));

  dumpData(DataSpirvBc, OutputFile);

  amd_comgr_(release_data(DataSpirv));
  amd_comgr_(release_data(DataSpirvBc));
  amd_comgr_(destroy_data_set(DataSetSpirv));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSpirv);
  return 0;
}
