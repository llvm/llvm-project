//===- logger-redirect.c ------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Driver for AMD_COMGR_REDIRECT_LOGS: enables per-action logging, runs one
// compile-to-BC action, and writes the caller's returned comgr.log to a file.
// With AMD_COMGR_REDIRECT_LOGS set by the test, this confirms redirection
// copies logs to the extra destination without moving them away from the
// caller's returned log.
//
// Usage: logger-redirect <source.cl> <ocl-version> <returned-log-outfile>

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc != 4)
    fail("usage: %s <source.cl> <ocl-version> <returned-log-outfile>", argv[0]);

  const char *SourceFile = argv[1];
  const char *OpenCLVersionStr = argv[2];
  const char *ReturnedLogFile = argv[3];

  char *BufSource;
  size_t SizeSource = setBuf(SourceFile, &BufSource);

  amd_comgr_language_t OpenCLVersion;
  if (strcmp(OpenCLVersionStr, "1.2") == 0)
    OpenCLVersion = AMD_COMGR_LANGUAGE_OPENCL_1_2;
  else if (strcmp(OpenCLVersionStr, "2.0") == 0)
    OpenCLVersion = AMD_COMGR_LANGUAGE_OPENCL_2_0;
  else
    fail("unsupported OCL version: %s", OpenCLVersionStr);

  amd_comgr_data_t DataSource;
  amd_comgr_data_set_t DataSetIn, DataSetBc;
  amd_comgr_action_info_t DataAction;

  amd_comgr_(create_data_set(&DataSetIn));
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource));
  amd_comgr_(set_data(DataSource, SizeSource, BufSource));
  amd_comgr_(set_data_name(DataSource, "source1.cl"));
  amd_comgr_(data_set_add(DataSetIn, DataSource));

  amd_comgr_(create_action_info(&DataAction));
  amd_comgr_(action_info_set_language(DataAction, OpenCLVersion));
  amd_comgr_(action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900"));
  // Enable per-action logging so a comgr.log is returned to the caller
  // alongside whatever AMD_COMGR_REDIRECT_LOGS copies to its destination.
  amd_comgr_(action_info_set_logging(DataAction, true));

  amd_comgr_(create_data_set(&DataSetBc));
  amd_comgr_(do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, DataAction,
                       DataSetIn, DataSetBc));

  size_t Count;
  amd_comgr_(action_data_count(DataSetBc, AMD_COMGR_DATA_KIND_LOG, &Count));
  if (Count != 1)
    fail("expected 1 returned log object, got %zu", Count);

  amd_comgr_data_t DataLog;
  amd_comgr_(
      action_data_get_data(DataSetBc, AMD_COMGR_DATA_KIND_LOG, 0, &DataLog));
  dumpData(DataLog, ReturnedLogFile);

  amd_comgr_(release_data(DataLog));
  amd_comgr_(release_data(DataSource));
  amd_comgr_(destroy_data_set(DataSetIn));
  amd_comgr_(destroy_data_set(DataSetBc));
  amd_comgr_(destroy_action_info(DataAction));
  free(BufSource);
  return 0;
}
