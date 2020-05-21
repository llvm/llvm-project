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
  amd_comgr_data_t dataCL;
  amd_comgr_data_set_t dataSetCL, dataSetBC, dataSetAsm;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  const char *buf = "kernel void f() { volatile int x = 0; }";
  size_t size = strlen(buf);

  status = amd_comgr_create_data_set(&dataSetCL);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataCL);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataCL, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataCL, "empty.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetCL, dataCL);
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

  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetCL, dataSetBC);
  checkError(status, "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC");
  checkCount("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC", dataSetBC, AMD_COMGR_DATA_KIND_BC, 1);

  status = amd_comgr_create_data_set(&dataSetAsm);
  checkError(status, "amd_comgr_create_data_set");
  const char *options[] = {"-Rpass-analysis=prolog"};
  size_t count = sizeof(options) / sizeof(options[0]);
  status = amd_comgr_action_info_set_option_list(dataAction, options, count);
  checkError(status, "amd_comgr_action_info_set_option_list");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               dataAction, dataSetBC, dataSetAsm);
  checkError(status, "AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY");
  checkCount("AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY", dataSetAsm,
             AMD_COMGR_DATA_KIND_SOURCE, 1);

  checkLogs("AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY", dataSetAsm,
            "remark: <unknown>:0:0: 8 stack bytes in function "
            "[-Rpass-analysis=prologepilog]");

  status = amd_comgr_destroy_data_set(dataSetCL);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetAsm);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  status = amd_comgr_release_data(dataCL);
  checkError(status, "amd_comgr_release_data");
}
