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
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  size_t size1;
  char *buf1;
  amd_comgr_data_t dataIn1;
  amd_comgr_data_set_t dataSetIn, dataSetOut;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  // Read input file
  size1 = setBuf(TEST_OBJ_DIR "/source1.s", &buf1);

  // Create data object
  {
    printf("Test create input data set\n");

    status = amd_comgr_create_data_set(&dataSetIn);
    checkError(status, "amd_cogmr_create_data_set");

    // File 1
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataIn1);
    checkError(status, "amd_comgr_create_data");
    status = amd_comgr_set_data(dataIn1, size1, buf1);
    checkError(status, "amd_comgr_set_data");
    status = amd_comgr_set_data_name(dataIn1, "source1.s");
    checkError(status, "amd_comgr_set_data_name");
    status = amd_comgr_data_set_add(dataSetIn, dataIn1);
    checkError(status, "amd_cogmr_data_set_add");
  }

  {
    printf("Test create empty output data set\n");

    status = amd_comgr_create_data_set(&dataSetOut);
    checkError(status, "amd_cogmr_create_data_set");
  }

  {
    printf("Test action assemble\n");
    status = amd_comgr_create_action_info(&dataAction);
    checkError(status, "amd_comgr_create_action_info");
    amd_comgr_action_info_set_isa_name(dataAction, "amdgcn-amd-amdhsa--gfx803");
    checkError(status, "amd_comgr_action_info_set_language");
    status = amd_comgr_action_info_set_options(dataAction, "");
    checkError(status, "amd_comgr_action_info_set_options");
    status =
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                            dataAction, dataSetIn, dataSetOut);
    checkError(status, "amd_comgr_do_action");
  }

  {
    printf("Test action outputs\n");
    // There should be two output data object
    size_t count;
    status = amd_comgr_action_data_count(
        dataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
    checkError(status, "amd_comgr_action_data_count");
    if (count == 1)
      printf("Passed, output 1 relocatable object\n");
    else {
      printf("Failed, output %ld relocatable objects (should output 1)\n",
             count);
      exit(1);
    }
  }

  {
    printf("Cleanup ...\n");
    status = amd_comgr_destroy_data_set(dataSetIn);
    checkError(status, "amd_comgr_destroy_data_set");
    status = amd_comgr_destroy_data_set(dataSetOut);
    checkError(status, "amd_comgr_destroy_data_set");
    status = amd_comgr_destroy_action_info(dataAction);
    checkError(status, "amd_comgr_destroy_action_info");
    status = amd_comgr_release_data(dataIn1);
    checkError(status, "amd_comgr_release_data");
    free(buf1);
  }

  return 0;
}
