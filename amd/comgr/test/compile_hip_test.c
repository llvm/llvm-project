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
  char *bufSource1;
  size_t sizeSource1;
  amd_comgr_data_t dataSource1;
  amd_comgr_data_set_t dataSetIn, dataSetFatBin;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;
  size_t count;
  const char *options[] = {"--amdgpu-target=gfx900", "-hip-path",
                           "/opt/rocm/hip"};
  size_t optionsCount = sizeof(options) / sizeof(options[0]);

  sizeSource1 = setBuf(TEST_OBJ_DIR "/source1.hip", &bufSource1);

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource1);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource1, sizeSource1, bufSource1);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource1, "source1.hip");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource1);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status =
      amd_comgr_action_info_set_language(dataAction, AMD_COMGR_LANGUAGE_HIP);
  checkError(status, "amd_comgr_action_info_set_language");
  status =
      amd_comgr_action_info_set_option_list(dataAction, options, optionsCount);
  checkError(status, "amd_comgr_action_info_set_option_list");

  status = amd_comgr_create_data_set(&dataSetFatBin);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN,
                               dataAction, dataSetIn, dataSetFatBin);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetFatBin,
                                       AMD_COMGR_DATA_KIND_FATBIN, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN Failed: "
           "produced %zu fab binaries (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_release_data(dataSource1);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetFatBin);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  free(bufSource1);
}
