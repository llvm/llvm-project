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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

int main(int argc, char *argv[]) {
  const char *bufInclude1 = "int x = 1;";
  size_t sizeInclude1 = strlen(bufInclude1);
  const char *bufInclude2 = "int y = 1;";
  size_t sizeInclude2 = strlen(bufInclude2);
  const char *bufInclude3 = "int z = 1;";
  size_t sizeInclude3 = strlen(bufInclude3);
  const char *bufSource = "#include \"subdir/header1.h\"\n#include \"sub/dir/header2.h\"\n#include \"sub/dir/header3.h\"";
  size_t sizeSource = strlen(bufSource);

  amd_comgr_data_t dataSource, dataInclude1, dataInclude2, dataInclude3;
  amd_comgr_data_set_t dataSetIn, dataSetPreproc;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource, sizeSource, bufSource);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource, "source.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &dataInclude1);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataInclude1, sizeInclude1, bufInclude1);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataInclude1, "subdir/header1.h");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataInclude1);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &dataInclude2);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataInclude2, sizeInclude2, bufInclude2);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataInclude2, "sub/dir/header2.h");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataInclude2);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &dataInclude3);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataInclude3, sizeInclude3, bufInclude3);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataInclude3, "sub/dir/header3.h");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataInclude3);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_language(dataAction, AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(dataAction, "amdgcn-amd-amdhsa--gfx803");
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_options(dataAction, "");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_create_data_set(&dataSetPreproc);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                               dataAction, dataSetIn, dataSetPreproc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetPreproc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  status = amd_comgr_release_data(dataInclude3);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataInclude2);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataInclude1);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataSource);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
}
