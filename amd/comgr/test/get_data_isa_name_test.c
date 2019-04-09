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

void testIsaName(amd_comgr_data_t data, const char *expectedIsaName) {
  size_t expectedSize = strlen(expectedIsaName) + 1;

  size_t size;
  char *isaName = NULL;
  amd_comgr_status_t status;

  status = amd_comgr_get_data_isa_name(data, &size, isaName);
  checkError(status, "amd_comgr_get_data_isa_name");

  if (size != expectedSize) {
    printf("amd_comgr_get_data_isa_name failed: produced %zu (expected %zu)\n",
           size, expectedSize);
    exit(1);
  }

  isaName = malloc(size);
  if (!isaName) {
    printf("cannot allocate %zu bytes for isa_name\n", size);
    exit(1);
  }

  status = amd_comgr_get_data_isa_name(data, &size, isaName);
  checkError(status, "amd_comgr_get_data_isa_name");

  if (strcmp(isaName, expectedIsaName)) {
    printf(
        "amd_comgr_get_data_isa_name failed: produced '%s' (expected '%s')\n",
        isaName, expectedIsaName);
    exit(1);
  }

  free(isaName);
}

void compileAndTestIsaName(const char *expectedIsaName, const char *options) {
  char *bufSource;
  size_t sizeSource;
  amd_comgr_data_t dataSource, dataReloc, dataExec;
  amd_comgr_status_t status;
  amd_comgr_data_set_t dataSetIn, dataSetBC, dataSetLinked, dataSetReloc,
      dataSetExec;
  amd_comgr_action_info_t dataAction;

  sizeSource = setBuf(TEST_OBJ_DIR "/shared.cl", &bufSource);

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource, sizeSource, bufSource);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource, "shared.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_language(dataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(dataAction, expectedIsaName);
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_options(dataAction, options);
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetIn, dataSetBC);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&dataSetLinked);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, dataAction,
                               dataSetBC, dataSetLinked);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&dataSetReloc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               dataAction, dataSetLinked, dataSetReloc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_get_data(
      dataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &dataReloc);
  checkError(status, "amd_comgr_action_data_get_data");

  status = amd_comgr_create_data_set(&dataSetExec);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               dataAction, dataSetReloc, dataSetExec);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_get_data(
      dataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &dataExec);
  checkError(status, "amd_comgr_action_data_get_data");

  testIsaName(dataReloc, expectedIsaName);
  testIsaName(dataExec, expectedIsaName);

  status = amd_comgr_release_data(dataSource);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataReloc);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataExec);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetLinked);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetReloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetExec);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  free(bufSource);
}

int main(int argc, char *argv[]) {
  size_t isaCount;
  amd_comgr_status_t status;

  status = amd_comgr_get_isa_count(&isaCount);
  checkError(status, "amd_comgr_get_isa_count");

  for (int i = 0; i < isaCount; i++) {
    const char *isaName;
    status = amd_comgr_get_isa_name(i, &isaName);
    checkError(status, "amd_comgr_get_isa_name");

    // Test object code v2.
    compileAndTestIsaName(isaName, "-mno-code-object-v3");
    // Test object code v3.
    compileAndTestIsaName(isaName, "-mcode-object-v3");
  }
}
