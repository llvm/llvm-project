/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 *******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  char *BufSource1;
  size_t SizeSource1;
  amd_comgr_data_t DataSource1;
  amd_comgr_data_set_t DataSetIn, DataSetFatBin;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;
  size_t Count;
  const char *Options[] = {"--amdgpu-target=gfx900"};
  size_t OptionsCount = sizeof(Options) / sizeof(Options[0]);

  SizeSource1 = setBuf(TEST_OBJ_DIR "/source1.hip", &BufSource1);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource1, SizeSource1, BufSource1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource1, "source1.hip");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource1);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status =
      amd_comgr_action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status =
      amd_comgr_action_info_set_option_list(DataAction, Options, OptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetFatBin);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN,
                               DataAction, DataSetIn, DataSetFatBin);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_count(DataSetFatBin,
                                       AMD_COMGR_DATA_KIND_FATBIN, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN Failed: "
           "produced %zu fab binaries (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_release_data(DataSource1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetFatBin);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource1);
}
