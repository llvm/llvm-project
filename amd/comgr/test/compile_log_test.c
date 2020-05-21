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
  amd_comgr_data_t dataCL, dataASM, dataBC, dataReloc;
  amd_comgr_data_set_t dataSetOut, dataSetCL, dataSetASM, dataSetBC,
      dataSetReloc;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  size_t count;
  const char *buf = "invalid";
  size_t size = strlen(buf);

  status = amd_comgr_create_data_set(&dataSetCL);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataCL);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataCL, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataCL, "invalid.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetCL, dataCL);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data_set(&dataSetASM);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataASM);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataASM, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataASM, "invalid.s");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetASM, dataASM);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &dataBC);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataBC, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataBC, "invalid.bc");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetBC, dataBC);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data_set(&dataSetReloc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataReloc);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataReloc, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataReloc, "invalid.o");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetReloc, dataReloc);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_language(dataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(dataAction,
                                              "amdgcn-amd-amdhsa--gfx803");
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_logging(dataAction, true);
  checkError(status, "amd_comgr_action_info_set_logging");

  // AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetCL, dataSetOut);
  checkLogs("COMPILE_SOURCE_TO_BC", dataSetOut,
            "error: unknown type name 'invalid'");
  checkLogs("COMPILE_SOURCE_TO_BC", dataSetOut,
            "2 errors generated.");

  status =
      amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu LOG objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_LINK_BC_TO_BC

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, dataAction,
                               dataSetBC, dataSetOut);
  checkLogs("LINK_BC_TO_BC", dataSetOut, "error: expected top-level entity");

  status =
      amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_LINK_BC_TO_BC Failed: "
           "produced %zu LOG objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                               dataAction, dataSetASM, dataSetOut);
  checkLogs("ASSEMBLE_SOURCE_TO_RELOCATABLE", dataSetOut,
            "error: invalid instruction");

  status =
      amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               dataAction, dataSetBC, dataSetOut);
  checkLogs("CODEGEN_BC_TO_RELOCATABLE", dataSetOut,
            "error: expected top-level entity");
  checkLogs("CODEGEN_BC_TO_RELOCATABLE", dataSetOut,
            "1 error generated.");

  status =
      amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               dataAction, dataSetReloc, dataSetOut);
  checkLogs("LINK_RELOCATABLE_TO_EXECUTABLE", dataSetOut, "unexpected EOF");

  status =
      amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG, &count);
  checkError(status, "amd_comgr_action_data_count");

  if (count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  status = amd_comgr_release_data(dataCL);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataASM);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataBC);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataReloc);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetCL);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetASM);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetReloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
}
