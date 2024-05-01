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
