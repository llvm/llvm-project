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
  const char *BufInclude1 = "int x = 1;";
  size_t SizeInclude1 = strlen(BufInclude1);
  const char *BufInclude2 = "int y = 1;";
  size_t SizeInclude2 = strlen(BufInclude2);
  const char *BufInclude3 = "int z = 1;";
  size_t SizeInclude3 = strlen(BufInclude3);
  const char *BufSource =
      "#include \"subdir/header1.h\"\n#include \"sub/dir/header2.h\"\n#include "
      "\"sub/dir/header3.h\"";
  size_t SizeSource = strlen(BufSource);

  amd_comgr_data_t DataSource, DataInclude1, DataInclude2, DataInclude3;
  amd_comgr_data_set_t DataSetIn, DataSetPreproc;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "source.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude1);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude1, SizeInclude1, BufInclude1);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude1, "subdir/header1.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude1);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude2);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude2, SizeInclude2, BufInclude2);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude2, "sub/dir/header2.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude2);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude3);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude3, SizeInclude3, BufInclude3);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude3, "sub/dir/header3.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude3);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  Status = amd_comgr_create_data_set(&DataSetPreproc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                               DataAction, DataSetIn, DataSetPreproc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_destroy_data_set(DataSetPreproc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_release_data(DataInclude3);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude2);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude1);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
}
