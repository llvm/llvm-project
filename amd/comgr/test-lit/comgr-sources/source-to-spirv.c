//===- source-to-spirv.c --------------------------------------------------===//
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
  amd_comgr_data_set_t DataSetSource, DataSetSpirv;
  amd_comgr_action_info_t DataAction;
  size_t Count;

  if (argc != 3) {
    fprintf(stderr, "Usage: source-to-spirv file.hip file.spv\n");
    exit(1);
  }

  SizeSource = setBuf(argv[1], &BufSource);

  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, SizeSource, BufSource));
  amd_comgr_(set_data_name(DataSource, "file.hip"));

  amd_comgr_(create_data_set(&DataSetSource));
  amd_comgr_(data_set_add(DataSetSource, DataSource));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP));

  amd_comgr_(create_data_set(&DataSetSpirv));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_SPIRV,
                       DataAction, DataSetSource, DataSetSpirv));

  amd_comgr_data_t DataSpirv;
  amd_comgr_(action_data_get_data(DataSetSpirv, AMD_COMGR_DATA_KIND_SPIRV,
                                  0, &DataSpirv));
  dumpData(DataSpirv, argv[2]);

  amd_comgr_(release_data(DataSource));
  amd_comgr_(destroy_data_set(DataSetSource));
  amd_comgr_(destroy_data_set(DataSetSpirv));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSource);
}

