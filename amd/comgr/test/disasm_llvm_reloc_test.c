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
  size_t size1, size2;
  char *buf1, *buf2;
  amd_comgr_data_t dataIn1, dataIn2, dataOut1, dataOut2;
  amd_comgr_data_set_t dataSetIn, dataSetOut;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  // Read input file
  size1 = setBuf(TEST_OBJ_DIR "/reloc1.o", &buf1);
  size2 = setBuf(TEST_OBJ_DIR "/reloc2.o", &buf2);

  // Create data object
  {
    printf("Test create input data set\n");

    status = amd_comgr_create_data_set(&dataSetIn);
    checkError(status, "amd_cogmr_create_data_set");

    // File 1
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn1);
    checkError(status, "amd_comgr_create_data");
    status = amd_comgr_set_data(dataIn1, size1, buf1);
    checkError(status, "amd_comgr_set_data");
    status = amd_comgr_set_data_name(dataIn1, "DO_IN1");
    checkError(status, "amd_comgr_set_data_name");
    status = amd_comgr_data_set_add(dataSetIn, dataIn1);
    checkError(status, "amd_cogmr_data_set_add");

    // File 2
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn2);
    checkError(status, "amd_comgr_create_data_2");
    status = amd_comgr_set_data(dataIn2, size2, buf2);
    checkError(status, "amd_comgr_set_data");
    status = amd_comgr_set_data_name(dataIn2, "DO_IN2");
    checkError(status, "amd_comgr_set_data_name_2");
    status = amd_comgr_data_set_add(dataSetIn, dataIn2);
    checkError(status, "amd_cogmr_data_set_add_2");
  }

  {
    printf("Test create empty output data set\n");

    status = amd_comgr_create_data_set(&dataSetOut);
    checkError(status, "amd_cogmr_create_data_set");
  }

  {
    printf("Test create action info\n");

    status = amd_comgr_create_action_info(&dataAction);
    checkError(status, "amd_comgr_create_action_info");
    status = amd_comgr_action_info_set_isa_name(dataAction,
                                                "amdgcn-amd-amdhsa--gfx803");
    checkError(status, "amd_comgr_action_info_set_isa_name");

    // Do disassembly action
    status =
        amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE,
                            dataAction, dataSetIn, dataSetOut);
    checkError(status, "amd_comgr_do_action");

    status = amd_comgr_destroy_data_set(dataSetIn);
    checkError(status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Test action outputs\n");
    // There should be two output data object
    size_t count;
    status = amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE,
                                         &count);
    checkError(status, "amd_comgr_action_data_count");
    if (count == 2)
      printf("Passed, output data object returned = 2\n");
    else
      printf("Failed, Output data object returned = %ld\n", count);

    // Retrieve the result data object 1 from dataSetOut
    status = amd_comgr_action_data_get_data(
        dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &dataOut1);
    checkError(status, "amd_comgr_action_data_get_data");
    status = amd_comgr_get_data(dataOut1, &count, NULL);
    checkError(status, "amd_comgr_get_data");
    char *bytes = (char *)malloc(count);
    status = amd_comgr_get_data(dataOut1, &count, bytes);
    checkError(status, "amd_comgr_get_data");

    printf("Output = \n");
    for (size_t i = 0; i < count; i++)
      printf("%c", bytes[i]);
    free(bytes);

    // Retrieve the result data object 2 from dataSetOut
    status = amd_comgr_action_data_get_data(
        dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 1, &dataOut2);
    checkError(status, "amd_comgr_action_data_get_data");
    status = amd_comgr_get_data(dataOut2, &count, NULL);
    checkError(status, "amd_comgr_get_data");
    char *bytes2 = (char *)malloc(count);
    status = amd_comgr_get_data(dataOut2, &count, bytes2);
    checkError(status, "amd_comgr_get_data");

    printf("Output = \n");
    for (size_t i = 0; i < count; i++)
      printf("%c", bytes2[i]);
    free(bytes2);

    status = amd_comgr_destroy_data_set(dataSetOut);
    checkError(status, "amd_comgr_destroy_data_set");
  }

  {
    printf("Cleanup ...\n");
    status = amd_comgr_destroy_action_info(dataAction);
    checkError(status, "amd_comgr_destroy_action_info");
    status = amd_comgr_release_data(dataIn1);
    checkError(status, "amd_comgr_release_data");
    status = amd_comgr_release_data(dataIn2);
    checkError(status, "amd_comgr_release_data");
    status = amd_comgr_release_data(dataOut1);
    checkError(status, "amd_comgr_release_data");
    status = amd_comgr_release_data(dataOut2);
    checkError(status, "amd_comgr_release_data");
    free(buf1);
    free(buf2);
  }

  return 0;
}
