//===- status-string.c ----------------------------------------------------===//
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
  const char *StatusString = malloc(sizeof(char) * 100);
  amd_comgr_(status_string(AMD_COMGR_STATUS_SUCCESS, &StatusString));
  if (strcmp(StatusString, "SUCCESS"))
    fail("incorrect status: expected 'SUCCESS', saw '%s'", StatusString);

  amd_comgr_(status_string(AMD_COMGR_STATUS_ERROR, &StatusString));
  if (strcmp(StatusString, "ERROR"))
    fail("incorrect status: expected 'ERROR', saw '%s'", StatusString);

  amd_comgr_(status_string(AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT,
                           &StatusString));
  if (strcmp(StatusString, "INVALID_ARGUMENT")) {
    fail("incorrect status: expected 'INVALID_ARGUMENT', saw '%s'",
         StatusString);
  }

  amd_comgr_(status_string(AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES,
                           &StatusString));
  if (strcmp(StatusString, "OUT_OF_RESOURCES")) {
    fail("incorrect status: expected 'OUT_OF_RESOURCES', saw '%s'",
         StatusString);
  }

  fail_amd_comgr_(status_string(-1, &StatusString));
  return 0;
}
