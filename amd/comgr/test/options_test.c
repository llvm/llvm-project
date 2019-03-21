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

void testFlat(amd_comgr_action_info_t actionInfo, const char *options) {
  amd_comgr_status_t status;

  status = amd_comgr_action_info_set_options(actionInfo, options);
  checkError(status, "amd_comgr_action_info_set_options");

  size_t size;
  status = amd_comgr_action_info_get_options(actionInfo, &size, NULL);
  checkError(status, "amd_comgr_action_info_get_options");

  char *retOptions = calloc(size, sizeof(char));
  status = amd_comgr_action_info_get_options(actionInfo, &size, retOptions);
  checkError(status, "amd_comgr_action_info_get_options");

  if (strcmp(options, retOptions))
    fail("incorrect options string: expected '%s', saw '%s'", options,
         retOptions);

  free(retOptions);
}

void testFlats() {
  amd_comgr_action_info_t actionInfo;
  amd_comgr_status_t status;
  size_t count;

  status = amd_comgr_create_action_info(&actionInfo);
  checkError(status, "amd_comgr_create_action_info");

  const char *options[] = {"foo", "foo bar", "bar baz qux",
                           "aaaaaaaaaaaaaaaaaaaaa"};
  size_t optionsCount = sizeof(options) / sizeof(options[0]);

  for (size_t i = 0; i < optionsCount; ++i)
    testFlat(actionInfo, options[i]);

  status = amd_comgr_destroy_action_info(actionInfo);
  checkError(status, "amd_comgr_destroy_action_info");
}

void testList(amd_comgr_action_info_t actionInfo, const char *options[],
              size_t count) {
  size_t actualCount;
  amd_comgr_status_t status;

  status = amd_comgr_action_info_set_option_list(actionInfo, options, count);
  checkError(status, "amd_comgr_action_info_set_option_list");

  status =
      amd_comgr_action_info_get_option_list_count(actionInfo, &actualCount);
  checkError(status, "amd_comgr_action_info_get_option_list_count");

  if (count != actualCount)
    fail("incorrect option count: expected %zu, saw %zu", count, actualCount);

  for (size_t i = 0; i < count; ++i) {
    size_t size;
    status =
        amd_comgr_action_info_get_option_list_item(actionInfo, i, &size, NULL);
    checkError(status, "amd_comgr_action_info_get_option_list_item");
    char *option = calloc(size, sizeof(char));
    status = amd_comgr_action_info_get_option_list_item(actionInfo, i, &size,
                                                        option);
    checkError(status, "amd_comgr_action_info_get_option_list_item");
    if (strcmp(options[i], option))
      fail("incorrect option string: expected '%s', saw '%s'", options[i],
           option);
    free(option);
  }
}

void testLists() {
  amd_comgr_action_info_t actionInfo;
  amd_comgr_status_t status;
  size_t count;

  status = amd_comgr_create_action_info(&actionInfo);
  checkError(status, "amd_comgr_create_action_info");

  const char *options[] = {"foo", "bar", "bazqux", "aaaaaaaaaaaaaaaaaaaaa"};
  size_t optionsCount = sizeof(options) / sizeof(options[0]);

  for (size_t i = 0; i <= optionsCount; ++i)
    for (size_t j = 0; i + j <= optionsCount; ++j)
      testList(actionInfo, options + i, j);

  status = amd_comgr_destroy_action_info(actionInfo);
  checkError(status, "amd_comgr_destroy_action_info");
}

void testMixed() {
  amd_comgr_action_info_t actionInfo;
  amd_comgr_status_t status;
  size_t size;

  status = amd_comgr_create_action_info(&actionInfo);
  checkError(status, "amd_comgr_create_action_info");

  // Confirm the default is the legacy flat options string.
  status = amd_comgr_action_info_get_options(actionInfo, &size, NULL);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    fail("expected new action_info to default to a flat options string, but "
         "amd_comgr_action_info_get_options fails");

  status = amd_comgr_action_info_get_option_list_count(actionInfo, &size);
  if (status != AMD_COMGR_STATUS_ERROR)
    fail("expected new action_info to default to a flat options string, but "
         "amd_comgr_action_info_get_option_list_count does not fail");

  // Confirm the inverse: if we set using a list, we should not be able to
  // access as if it were flat.
  status = amd_comgr_action_info_set_option_list(actionInfo, NULL, 0);
  checkError(status, "amd_comgr_action_info_set_option_list");

  status = amd_comgr_action_info_get_options(actionInfo, &size, NULL);
  if (status != AMD_COMGR_STATUS_ERROR)
    fail("amd_comgr_action_info_get_options does not fail with list options");

  status = amd_comgr_action_info_get_option_list_count(actionInfo, &size);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    fail("amd_comgr_action_info_get_option_list_count fails with list options");

  // Also confirm we can switch back to flat.
  status = amd_comgr_action_info_set_options(actionInfo, "");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_action_info_get_options(actionInfo, &size, NULL);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    fail("amd_comgr_action_info_get_options fails with flat options string");

  status = amd_comgr_action_info_get_option_list_count(actionInfo, &size);
  if (status != AMD_COMGR_STATUS_ERROR)
    fail("amd_comgr_action_info_get_option_list_count does not fail with flat "
         "options string");

  status = amd_comgr_destroy_action_info(actionInfo);
  checkError(status, "amd_comgr_destroy_action_info");
}

void testFlatSplitting() {
  char *bufSource, *bufInclude;
  size_t sizeSource, sizeInclude;
  amd_comgr_data_t dataSource, dataInclude;
  amd_comgr_data_set_t dataSetIn, dataSetBC, dataSetDevLibs;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  sizeSource = setBuf(TEST_OBJ_DIR "/source1.cl", &bufSource);
  sizeInclude = setBuf(TEST_OBJ_DIR "/include-a.h", &bufInclude);

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &dataSource);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataSource, sizeSource, bufSource);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataSource, "source1.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataSource);
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

  // Confirm we get space-delimited for non-device-libs actions

  // Check with empty string
  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction, "");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetIn, dataSetBC);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");

  // Check with a single option
  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction, "-O3");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetIn, dataSetBC);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");

  // Check with a multiple options
  status = amd_comgr_create_data_set(&dataSetBC);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction,
                                             "-mllvm -amdgpu-early-inline-all");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               dataAction, dataSetIn, dataSetBC);
  checkError(status, "amd_comgr_do_action");

  // Confirm we get comma-delimited for the device-libs action

  // Check with empty string
  status = amd_comgr_create_data_set(&dataSetDevLibs);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction, "");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               dataAction, dataSetBC, dataSetDevLibs);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetDevLibs);
  checkError(status, "amd_comgr_destroy_data_set");

  // Check with a single option
  status = amd_comgr_create_data_set(&dataSetDevLibs);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_action_info_set_options(dataAction, "finite_only");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               dataAction, dataSetBC, dataSetDevLibs);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetDevLibs);
  checkError(status, "amd_comgr_destroy_data_set");

  // Check with multiple options
  status = amd_comgr_create_data_set(&dataSetDevLibs);
  checkError(status, "amd_comgr_create_data_set");

  status =
      amd_comgr_action_info_set_options(dataAction, "finite_only,unsafe_math");
  checkError(status, "amd_comgr_action_info_set_options");

  status = amd_comgr_do_action(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                               dataAction, dataSetBC, dataSetDevLibs);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_release_data(dataSource);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(dataInclude);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetBC);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(dataSetDevLibs);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  free(bufSource);
  free(bufInclude);
}

int main(int argc, char *argv[]) {
  testFlats();
  testLists();
  testMixed();
  testFlatSplitting();
}
