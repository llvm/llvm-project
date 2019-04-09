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
  char *bufSource1, *bufSource2, *bufInclude;
  size_t sizeSource1, sizeSource2, sizeInclude;
  amd_comgr_data_t dataSource1, dataSource2, dataInclude;
  amd_comgr_data_set_t dataSetIn, dataSetPreproc, dataSetBC, dataSetLinked,
      dataSetAsm, dataSetReloc, dataSetExec;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;
  size_t count;

  sizeSource1 = setBuf(TEST_OBJ_DIR "/source1.cl", &bufSource1);
  sizeSource2 = setBuf(TEST_OBJ_DIR "/source2.cl", &bufSource2);
  sizeInclude = setBuf(TEST_OBJ_DIR "/include-a.h", &bufInclude);

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource1);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource1, sizeSource1, bufSource1);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource1, "source1.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource1);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource2);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource2, sizeSource2, bufSource2);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource2, "source2.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource2);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &dataInclude);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataInclude, sizeInclude, bufInclude);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataInclude, "include-a.h");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataInclude);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_language(dataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(dataAction,
                                              "amdgcn-amd-amdhsa--gfx803");
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_options(dataAction,
                                             "-mllvm -amdgpu-early-inline-all");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_create_data_set(&dataSetPreproc);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                               dataAction, dataSetIn, dataSetPreproc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetPreproc,
                                       AMD_COMGR_DATA_KIND_SOURCE, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 2) {
    printf("AMD_COMGR_ACTION_PREPROCESS_SOURCE_TO_SOURCE Failed: "
           "produced %zu source objects (expected 2)\n",
           count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetPreproc, dataSetBC);
  checkError(status, "amd_comgr_do_action");

  status =
      amd_comgr_action_data_count(dataSetBC, AMD_COMGR_DATA_KIND_BC, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 2) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu BC objects (expected 2)\n",
           count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetLinked);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction,
                                             "-mllvm -amdgpu-early-inline-all");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, dataAction,
                               dataSetBC, dataSetLinked);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetLinked, AMD_COMGR_DATA_KIND_BC,
                                       &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_LINK_BC_TO_BC Failed: "
           "produced %zu BC objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetAsm);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               dataAction, dataSetLinked, dataSetAsm);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetAsm, AMD_COMGR_DATA_KIND_SOURCE,
                                       &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY Failed: "
           "produced %zu source objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetReloc);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                               dataAction, dataSetAsm, dataSetReloc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetReloc,
                                       AMD_COMGR_DATA_KIND_RELOCATABLE, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE Failed: "
           "produced %zu relocatable objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_create_data_set(&dataSetExec);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction, "");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               dataAction, dataSetReloc, dataSetExec);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_count(dataSetExec,
                                       AMD_COMGR_DATA_KIND_EXECUTABLE, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu executable objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_release_data(dataSource1);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataSource2);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataInclude);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetPreproc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetLinked);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetAsm);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetReloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetExec);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  free(bufSource1);
  free(bufSource2);
  free(bufInclude);
}
