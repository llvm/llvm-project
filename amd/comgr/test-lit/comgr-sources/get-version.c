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

  if (*Major != AMD_COMGR_INTERFACE_VERSION_MAJOR ||
      *Minor != AMD_COMGR_INTERFACE_VERSION_MINOR)
    fail("incorrect version: expected %d.%d, saw %zu, %zu",
         AMD_COMGR_INTERFACE_VERSION_MAJOR, AMD_COMGR_INTERFACE_VERSION_MINOR,
         *Major, *Minor);

  free(Major);
  free(Minor);
  return 0;
}
