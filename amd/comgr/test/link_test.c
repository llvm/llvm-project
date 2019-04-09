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
  size_t count;
  amd_comgr_data_t dataIn1, dataIn2;
  amd_comgr_data_set_t dataSetIn, dataSetOutReloc, dataSetOutExec;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  // Read input file
  size1 = setBuf(TEST_OBJ_DIR "/reloc1.o", &buf1);
  size2 = setBuf(TEST_OBJ_DIR "/reloc2.o", &buf2);

  // Create data object
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

  status = amd_comgr_create_data_set(&dataSetOutReloc);
  checkError(status, "amd_cogmr_create_data_set");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  amd_comgr_action_info_set_isa_name(dataAction, "amdgcn-amd-amdhsa--gfx803");
  checkError(status, "amd_comgr_action_info_set_language");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE,
                               dataAction, dataSetIn, dataSetOutReloc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetOutReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
  checkError(status, "amd_comgr_action_data_count");
  if (count != 1) {
    printf("Failed, output %ld relocatable objects (should output 1)\n", count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetOutExec);
  checkError(status, "amd_cogmr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               dataAction, dataSetIn, dataSetOutExec);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetOutExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &count);
  checkError(status, "amd_comgr_action_data_count");
  if (count != 1) {
    printf("Failed, output %ld executable objects (should output 1)\n", count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetOutReloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetOutExec);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  status = amd_comgr_release_data(dataIn1);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataIn2);
  checkError(status, "amd_comgr_release_data");
  free(buf1);
  free(buf2);

  return 0;
}
