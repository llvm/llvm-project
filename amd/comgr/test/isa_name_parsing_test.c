//===- isa_name_parsing_test.c --------------------------------------------===//
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

void parseIsaName(amd_comgr_action_info_t DataAction, const char *IsaName,
                  amd_comgr_status_t ExpectedStatus) {
  amd_comgr_status_t TrueStatus =
      amd_comgr_action_info_set_isa_name(DataAction, IsaName);
  if (TrueStatus != ExpectedStatus) {
    amd_comgr_status_t Status;
    const char *TrueStatusString, *ExpectedStatusString;
    Status = amd_comgr_status_string(TrueStatus, &TrueStatusString);
    checkError(Status, "amd_comgr_status_string");
    Status = amd_comgr_status_string(ExpectedStatus, &ExpectedStatusString);
    checkError(Status, "amd_comgr_status_string");
    printf("Parsing \"%s\" resulted in \"%s\"; expected \"%s\"\n", IsaName,
           TrueStatusString, ExpectedStatusString);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;
  amd_comgr_action_info_t dataAction;

  Status = amd_comgr_create_action_info(&dataAction);
  checkError(Status, "amd_comgr_create_action_info");

#define PARSE_VALID_ISA_NAME(name)                                             \
  parseIsaName(dataAction, name, AMD_COMGR_STATUS_SUCCESS)
#define PARSE_INVALID_ISA_NAME(name)                                           \
  parseIsaName(dataAction, name, AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT)

  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack-");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:sramecc-");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack+:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack-:sramecc+");
  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx908:xnack-:sramecc-");

  PARSE_VALID_ISA_NAME("amdgcn-amd-amdhsa--gfx1010:xnack+");
  PARSE_VALID_ISA_NAME("");
  PARSE_VALID_ISA_NAME(NULL);

  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx801:xnack+:sramecc+");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803:::");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa-opencl-gfx803");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa-gfx803");
  PARSE_INVALID_ISA_NAME("gfx803");
  PARSE_INVALID_ISA_NAME(" amdgcn-amd-amdhsa--gfx803");
  PARSE_INVALID_ISA_NAME(" amdgcn-amd-amdhsa--gfx803 ");
  PARSE_INVALID_ISA_NAME("amdgcn-amd-amdhsa--gfx803 ");
  PARSE_INVALID_ISA_NAME("   amdgcn-amd-amdhsa--gfx803  ");

  Status = amd_comgr_destroy_action_info(dataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
}
