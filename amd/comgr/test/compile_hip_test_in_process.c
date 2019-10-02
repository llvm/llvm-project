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

int main(int Argc, char *Argv[]) {
  char *bufSource;
  size_t sizeSource;
  amd_comgr_data_t DataSrc;
  amd_comgr_data_set_t DataSetSrc, DataSetBc,
      DataSetDevLibs, DataSetLinkedBc, DataSetAsm,
      DataSetReloc, DataSetExec;
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t status;

  sizeSource = setBuf(TEST_OBJ_DIR "/source2.hip", &bufSource);

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSrc);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(DataSrc, sizeSource, bufSource);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(DataSrc, "source2.hip");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_create_data_set(&DataSetSrc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_data_set_add(DataSetSrc, DataSrc);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&ActionInfo);
  checkError(status, "amd_comgr_create_action_info");
  status =
      amd_comgr_action_info_set_language(ActionInfo, AMD_COMGR_LANGUAGE_HIP);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(
      ActionInfo, "amdgcn-amd-amdhsa--gfx906");
  checkError(status, "amd_comgr_action_info_set_isa_name");

  status = amd_comgr_create_data_set(&DataSetBc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               ActionInfo, DataSetSrc, DataSetBc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&DataSetDevLibs);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               ActionInfo, DataSetBc, DataSetDevLibs);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&DataSetLinkedBc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, ActionInfo,
                               DataSetDevLibs, DataSetLinkedBc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&DataSetAsm);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               ActionInfo, DataSetLinkedBc, DataSetAsm);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               ActionInfo, DataSetLinkedBc, DataSetReloc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&DataSetExec);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               ActionInfo, DataSetReloc, DataSetExec);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(status, "amd_comgr_destroy_action_info");
  status = amd_comgr_destroy_data_set(DataSetSrc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetDevLibs);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetLinkedBc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetAsm);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_release_data(DataSrc);
  checkError(status, "amd_comgr_release_data");

  free(bufSource);
}
