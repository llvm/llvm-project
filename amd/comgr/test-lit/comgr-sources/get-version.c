//===- get-version.c ------------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "common.h"

int main(int argc, char *argv[]) {

  size_t *Major = malloc(sizeof(size_t));
  size_t *Minor = malloc(sizeof(size_t));

  amd_comgr_get_version(Major, Minor);

  if (*Major != 3 || *Minor != 0)
    fail("incorrect version: expected 3.0, saw %zu, %zu", *Major, *Minor);

  free(Major);
  free(Minor);
  return 0;
}
