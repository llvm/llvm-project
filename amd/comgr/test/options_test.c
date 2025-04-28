//===- options_test.c -----------------------------------------------------===//
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

int main(int argc, char *argv[]) {
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
