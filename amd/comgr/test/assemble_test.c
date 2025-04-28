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
  size_t Size1;
  char *Buf1;
  amd_comgr_data_t DataIn1;
  amd_comgr_data_set_t DataSetIn, DataSetOut;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  // Read input file
  Size1 = setBuf(TEST_OBJ_DIR "/source1.s", &Buf1);

  // Create data object
  {
    printf("Test create input data set\n");

    Status = amd_comgr_create_data_set(&DataSetIn);
    checkError(Status, "amd_cogmr_create_data_set");

    // File 1
    Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataIn1);
    checkError(Status, "amd_comgr_create_data");
    Status = amd_comgr_set_data(DataIn1, Size1, Buf1);
    checkError(Status, "amd_comgr_set_data");
    Status = amd_comgr_set_data_name(DataIn1, "source1_no_extension");
    checkError(Status, "amd_comgr_set_data_name");
    Status = amd_comgr_data_set_add(DataSetIn, DataIn1);
    checkError(Status, "amd_cogmr_data_set_add");
  }

  {
    printf("Test create empty output data set\n");

    Status = amd_comgr_create_data_set(&DataSetOut);
    checkError(Status, "amd_cogmr_create_data_set");
  }

  {
    printf("Test action assemble\n");
    Status = amd_comgr_create_action_info(&DataAction);
    checkError(Status, "amd_comgr_create_action_info");
    amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx900");
    checkError(Status, "amd_comgr_action_info_set_language");
    Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
    checkError(Status, "amd_comgr_action_info_set_option_list");
    Status =
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                            DataAction, DataSetIn, DataSetOut);
    checkError(Status, "amd_comgr_do_action");
  }

  {
    printf("Test action outputs\n");
    // There should be two output data object
    size_t Count;
    Status = amd_comgr_action_data_count(
        DataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, &Count);
    checkError(Status, "amd_comgr_action_data_count");
    if (Count == 1) {
      printf("Passed, output 1 relocatable object\n");
    } else {
      printf("Failed, output %zd relocatable objects (should output 1)\n",
             Count);
      exit(1);
    }
  }

  {
    printf("Cleanup ...\n");
    Status = amd_comgr_destroy_data_set(DataSetIn);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_data_set(DataSetOut);
    checkError(Status, "amd_comgr_destroy_data_set");
    Status = amd_comgr_destroy_action_info(DataAction);
    checkError(Status, "amd_comgr_destroy_action_info");
    Status = amd_comgr_release_data(DataIn1);
    checkError(Status, "amd_comgr_release_data");
    free(Buf1);
  }

  return 0;
}
