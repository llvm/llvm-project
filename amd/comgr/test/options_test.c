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

void testFlat(amd_comgr_action_info_t ActionInfo, const char *Options) {
  amd_comgr_status_t Status;

  Status = amd_comgr_action_info_set_options(ActionInfo, Options);
  checkError(Status, "amd_comgr_action_info_set_options");

  size_t Size;
  Status = amd_comgr_action_info_get_options(ActionInfo, &Size, NULL);
  checkError(Status, "amd_comgr_action_info_get_options");

  char *RetOptions = calloc(Size, sizeof(char));
  Status = amd_comgr_action_info_get_options(ActionInfo, &Size, RetOptions);
  checkError(Status, "amd_comgr_action_info_get_options");

  if (strcmp(Options, RetOptions)) {
    fail("incorrect options string: expected '%s', saw '%s'", Options,
         RetOptions);
  }

  free(RetOptions);
}

void testFlats() {
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");

  const char *Options[] = {"foo", "foo bar", "bar baz qux",
                           "aaaaaaaaaaaaaaaaaaaaa"};
  size_t OptionsCount = sizeof(Options) / sizeof(Options[0]);

  for (size_t I = 0; I < OptionsCount; ++I) {
    testFlat(ActionInfo, Options[I]);
  }

  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
}

void testList(amd_comgr_action_info_t ActionInfo, const char *Options[],
              size_t Count) {
  size_t ActualCount;
  amd_comgr_status_t Status;

  Status = amd_comgr_action_info_set_option_list(ActionInfo, Options, Count);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status =
      amd_comgr_action_info_get_option_list_count(ActionInfo, &ActualCount);
  checkError(Status, "amd_comgr_action_info_get_option_list_count");

  if (Count != ActualCount) {
    fail("incorrect option count: expected %zu, saw %zu", Count, ActualCount);
  }

  for (size_t I = 0; I < Count; ++I) {
    size_t Size;
    Status =
        amd_comgr_action_info_get_option_list_item(ActionInfo, I, &Size, NULL);
    checkError(Status, "amd_comgr_action_info_get_option_list_item");
    char *Option = calloc(Size, sizeof(char));
    Status = amd_comgr_action_info_get_option_list_item(ActionInfo, I, &Size,
                                                        Option);
    checkError(Status, "amd_comgr_action_info_get_option_list_item");
    if (strcmp(Options[I], Option)) {
      fail("incorrect option string: expected '%s', saw '%s'", Options[I],
           Option);
    }
    free(Option);
  }
}

void testLists() {
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");

  const char *Options[] = {"foo", "bar", "bazqux", "aaaaaaaaaaaaaaaaaaaaa"};
  size_t OptionsCount = sizeof(Options) / sizeof(Options[0]);

  for (size_t I = 0; I <= OptionsCount; ++I) {
    for (size_t J = 0; I + J <= OptionsCount; ++J) {
      testList(ActionInfo, Options + I, J);
    }
  }

  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
}

void testMixed() {
  amd_comgr_action_info_t ActionInfo;
  amd_comgr_status_t Status;
  size_t Size;

  Status = amd_comgr_create_action_info(&ActionInfo);
  checkError(Status, "amd_comgr_create_action_info");

  // Confirm the default is the legacy flat options string.
  Status = amd_comgr_action_info_get_options(ActionInfo, &Size, NULL);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    fail("expected new action_info to default to a flat options string, but "
         "amd_comgr_action_info_get_options fails");
  }

  Status = amd_comgr_action_info_get_option_list_count(ActionInfo, &Size);
  if (Status != AMD_COMGR_STATUS_ERROR) {
    fail("expected new action_info to default to a flat options string, but "
         "amd_comgr_action_info_get_option_list_count does not fail");
  }

  // Confirm the inverse: if we set using a list, we should not be able to
  // access as if it were flat.
  Status = amd_comgr_action_info_set_option_list(ActionInfo, NULL, 0);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_action_info_get_options(ActionInfo, &Size, NULL);
  if (Status != AMD_COMGR_STATUS_ERROR) {
    fail("amd_comgr_action_info_get_options does not fail with list options");
  }

  Status = amd_comgr_action_info_get_option_list_count(ActionInfo, &Size);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    fail("amd_comgr_action_info_get_option_list_count fails with list options");
  }

  // Also confirm we can switch back to flat.
  Status = amd_comgr_action_info_set_options(ActionInfo, "");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_action_info_get_options(ActionInfo, &Size, NULL);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    fail("amd_comgr_action_info_get_options fails with flat options string");
  }

  Status = amd_comgr_action_info_get_option_list_count(ActionInfo, &Size);
  if (Status != AMD_COMGR_STATUS_ERROR) {
    fail("amd_comgr_action_info_get_option_list_count does not fail with flat "
         "options string");
  }

  Status = amd_comgr_destroy_action_info(ActionInfo);
  checkError(Status, "amd_comgr_destroy_action_info");
}

void testFlatSplitting() {
  char *BufSource, *BufInclude;
  size_t SizeSource, SizeInclude;
  amd_comgr_data_t DataSource, DataInclude;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetDevLibs;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  SizeSource = setBuf(TEST_OBJ_DIR "/source1.cl", &BufSource);
  SizeInclude = setBuf(TEST_OBJ_DIR "/include-macro.h", &BufInclude);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "source1.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_INCLUDE, &DataInclude);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataInclude, SizeInclude, BufInclude);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataInclude, "include-macro.h");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataInclude);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");

  // Confirm we get space-delimited for non-device-libs actions

  // Check with empty string
  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_options(DataAction, "");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action_compile_source_to_bc");

  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");

  // Check with a single option
  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_options(DataAction, "-O3");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action_compile_source_to_bc");

  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");

  // Check with a multiple options
  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_options(DataAction, "-mllvm --color");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action_compile_source_to_bc");

  // Confirm we get comma-delimited for the device-libs action

  // Check with empty string
  Status = amd_comgr_create_data_set(&DataSetDevLibs);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_options(DataAction, "");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               DataAction, DataSetBc, DataSetDevLibs);
  checkError(Status, "amd_comgr_do_action_add_device_libraries 0");

  Status = amd_comgr_destroy_data_set(DataSetDevLibs);
  checkError(Status, "amd_comgr_destroy_data_set");

  // Check with a single option
  Status = amd_comgr_create_data_set(&DataSetDevLibs);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_action_info_set_options(DataAction, "finite_only");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               DataAction, DataSetBc, DataSetDevLibs);
  checkError(Status, "amd_comgr_do_action_add_device_libraries 1");

  Status = amd_comgr_destroy_data_set(DataSetDevLibs);
  checkError(Status, "amd_comgr_destroy_data_set");

  // Check with multiple options
  Status = amd_comgr_create_data_set(&DataSetDevLibs);
  checkError(Status, "amd_comgr_create_data_set");

  Status =
      amd_comgr_action_info_set_options(DataAction, "finite_only,unsafe_math");
  checkError(Status, "amd_comgr_action_info_set_options");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               DataAction, DataSetBc, DataSetDevLibs);
  checkError(Status, "amd_comgr_do_action_add_device_libraries 2");

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataInclude);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetDevLibs);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource);
  free(BufInclude);
}

int main(int argc, char *argv[]) {
  testFlats();
  testLists();
  testMixed();
  testFlatSplitting();
}
