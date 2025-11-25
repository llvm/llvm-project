//===- compile-minimal-test.c ---------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  amd_comgr_action_info_t DataAction;
  amd_comgr_(create_action_info(&DataAction));

  // ---- set_language, get_language
  amd_comgr_language_t Language;
  amd_comgr_(
      action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_NONE));
  amd_comgr_(action_info_get_language(DataAction, &Language));
  if (Language != AMD_COMGR_LANGUAGE_NONE)
    fail("AMD_COMGR_LANGUAGE_NONE not returned!");

  amd_comgr_(
      action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_OPENCL_1_2));
  amd_comgr_(action_info_get_language(DataAction, &Language));
  if (Language != AMD_COMGR_LANGUAGE_OPENCL_1_2)
    fail("AMD_COMGR_LANGUAGE_OPENCL_1_2 not returned!");

  amd_comgr_(
    action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_OPENCL_2_0));
  amd_comgr_(action_info_get_language(DataAction, &Language));
  if (Language != AMD_COMGR_LANGUAGE_OPENCL_2_0)
    fail("AMD_COMGR_LANGUAGE_OPENCL_2_0 not returned!");

  amd_comgr_(
    action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_HIP));
  amd_comgr_(action_info_get_language(DataAction, &Language));
  if (Language != AMD_COMGR_LANGUAGE_HIP)
    fail("AMD_COMGR_LANGUAGE_HIP not returned!");

  amd_comgr_(
    action_info_set_language(DataAction, AMD_COMGR_LANGUAGE_LLVM_IR));
  amd_comgr_(action_info_get_language(DataAction, &Language));
  if (Language != AMD_COMGR_LANGUAGE_LLVM_IR)
    fail("AMD_COMGR_LANGUAGE_LLVM_IR not returned!");

  // ---- set_isa_name, get_isa_name
  // Tested in comgr/test/get_data_isa_name_test.c

  // ---- set_option_list, get_option_list_count, get_option_list_item
  const char *Options[] = {"foo", "bar", "bazqux", "aaaaaaaaaaaaaaaaaaaaa"};
  size_t OptionsCount = sizeof(Options) / sizeof(Options[0]);

  amd_comgr_(action_info_set_option_list(DataAction, Options, OptionsCount));

  size_t ActualCount;
  amd_comgr_(action_info_get_option_list_count(DataAction, &ActualCount));

  if (OptionsCount != ActualCount) {
    fail("incorrect option count: expected %zu, saw %zu", OptionsCount,
         ActualCount);
  }

  size_t Size;
  for (size_t I = 0; I < OptionsCount; ++I) {
    amd_comgr_(action_info_get_option_list_item(DataAction, I, &Size, NULL));

    char *Option = calloc(Size, sizeof(char));
    amd_comgr_(action_info_get_option_list_item(DataAction, I, &Size, Option));

    if (strcmp(Options[I], Option)) {
      fail("incorrect option string: expected '%s', saw '%s'", Options[I],
           Option);
    }
    free(Option);
  }

  fail_amd_comgr_(action_info_get_option_list_item(DataAction, OptionsCount,
                                                   &Size, NULL));
  fail_amd_comgr_(action_info_get_option_list_count(DataAction, NULL));
  fail_amd_comgr_(action_info_get_option_list_item(DataAction, 0, NULL, NULL));

  // ---- set_bundle_entry_ids, get_bundle_entry_id_count, get_bundle_entry_id
  // Tested in comgr/test/unbundle-hip-test.c

  // ---- set_working_directory_path, get_working_directory_path
  const char *Path = "/path/to/my/directory";
  amd_comgr_(action_info_set_working_directory_path(DataAction, Path));

  amd_comgr_(action_info_get_working_directory_path(DataAction, &Size,
                                                    NULL));
  char *GetPath = calloc(Size, sizeof(char));
  amd_comgr_(action_info_get_working_directory_path(DataAction, &Size,
                                                    GetPath));

  if (strcmp(Path, GetPath))
    fail("incorrect path string: expected '%s', saw '%s'", Path, GetPath);
  free(GetPath);

  // ---- set_logging, get_logging
  amd_comgr_(action_info_set_logging(DataAction, true));

  bool GetLogging;
  amd_comgr_(action_info_get_logging(DataAction, &GetLogging));

  if (!GetLogging)
    fail("incorrect logging boolean: expected 'true', saw 'false'");

  amd_comgr_(action_info_set_logging(DataAction, false));
  amd_comgr_(action_info_get_logging(DataAction, &GetLogging));

  if (GetLogging)
    fail("incorrect logging boolean: expected 'false', saw 'true'");

  // ---- set_device_lib_linking
  amd_comgr_(action_info_set_device_lib_linking(DataAction, true));
  amd_comgr_(action_info_set_device_lib_linking(DataAction, false));

  // ---- set_vfs
  amd_comgr_(action_info_set_vfs(DataAction, true));
  amd_comgr_(action_info_set_vfs(DataAction, false));

  amd_comgr_(destroy_action_info(DataAction));
  return 0;
}
