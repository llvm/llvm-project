//===- get_data_is_name_test.c --------------------------------------------===//
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

#define MAX_ISA_NAME_SIZE 1024

typedef enum {
  none,
  off,
  on,
  any
} feature_mode_t;

typedef struct {
  const char *IsaName;
  bool SrameccSupported;
  bool XnackSupported;
  bool NeedsCOV6;
} isa_features_t;

/* Features supported based on https://llvm.org/docs/AMDGPUUsage.html . */
static isa_features_t IsaFeatures[] = {
    // clang-format off
  //        ISA Name                     SRAMECC XNACK   NeedsCOV7
  {"amdgcn-amd-amdhsa--gfx600",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx601",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx602",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx700",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx701",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx702",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx703",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx704",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx705",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx801",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx802",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx803",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx805",          false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx810",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx900",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx902",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx904",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx906",          true,   true,   false},
  {"amdgcn-amd-amdhsa--gfx908",          true,   true,   false},
  {"amdgcn-amd-amdhsa--gfx909",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx90a",          true,   true,   false},
  {"amdgcn-amd-amdhsa--gfx90c",          false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx942",          true,   true,   false},
  {"amdgcn-amd-amdhsa--gfx950",          true,   true,   false},
  {"amdgcn-amd-amdhsa--gfx1010",         false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx1011",         false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx1012",         false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx1013",         false,  true,   false},
  {"amdgcn-amd-amdhsa--gfx1030",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1031",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1032",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1033",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1034",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1035",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1036",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1100",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1101",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1102",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1103",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1150",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1151",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1152",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1153",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1200",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1201",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1250",         false,  false,  false},
  {"amdgcn-amd-amdhsa--gfx1251",         false,  false,  false},

  {"amdgcn-amd-amdhsa--gfx9-generic",    false,  true,   true},
  {"amdgcn-amd-amdhsa--gfx9-4-generic",  true,   true,   true},
  {"amdgcn-amd-amdhsa--gfx10-1-generic", false,  true,   true},
  {"amdgcn-amd-amdhsa--gfx10-3-generic", false,  false,  true},
  {"amdgcn-amd-amdhsa--gfx11-generic",   false,  false,  true},
  {"amdgcn-amd-amdhsa--gfx12-generic",   false,  false,  true},
    // clang-format on
};

static size_t IsaFeaturesSize = sizeof(IsaFeatures) / sizeof(IsaFeatures[0]);

bool hasSubString(const char *String, const char *Sub) {
  return !strncmp(String, Sub, strlen(Sub));
}

bool getExpectedIsaName(unsigned CodeObjectVersion, const char *IsaName,
                        char *ExpectedIsaName, bool *NeedsCoV6) {
  char TokenizedIsaName[MAX_ISA_NAME_SIZE];

  strncpy(TokenizedIsaName, IsaName, MAX_ISA_NAME_SIZE);

  char *Token = strtok(TokenizedIsaName, ":");
  isa_features_t *Isa = NULL;
  for (size_t I = 0; I < IsaFeaturesSize; I++) {
    if (strncmp(Token, IsaFeatures[I].IsaName, MAX_ISA_NAME_SIZE) == 0) {
      Isa = &IsaFeatures[I];
      break;
    }
  }
  if (!Isa) {
    printf("The %s target is not supported by the test (update the "
           "isa_features table)\n",
           Token);
    exit(1);
  }

  *NeedsCoV6 = Isa->NeedsCOV6;
  strncpy(ExpectedIsaName, Isa->IsaName, MAX_ISA_NAME_SIZE);

  feature_mode_t Sramecc = any;
  feature_mode_t Xnack = any;

  Token = strtok(NULL, ":");
  while (Token != NULL) {
    if (strncmp(Token, "sramecc", strlen("sramecc")) == 0 &&
        Isa->SrameccSupported) {
      switch (Token[strlen("sramecc")]) {
      case '-':
        Sramecc = off;
        break;
      case '+':
        Sramecc = on;
        break;
      }
    }

    if (strncmp(Token, "xnack", strlen("xnack")) == 0 && Isa->XnackSupported) {
      switch (Token[strlen("xnack")]) {
      case '-':
        Xnack = off;
        break;
      case '+':
        Xnack = on;
        break;
      }
    }

    Token = strtok(NULL, ":");
  }

  switch (CodeObjectVersion) {
  case 4:
  case 5:
  case 6:
    // All ISA strings are valid.
    return true;

  default:
    printf("Code object V%u is not supported by the test (update the "
           "get_expected_isa_name)\n",
           CodeObjectVersion);
    exit(1);
  }

  strncpy(ExpectedIsaName, Isa->IsaName, MAX_ISA_NAME_SIZE);

  if (Isa->SrameccSupported && Sramecc != any) {
    strncat(ExpectedIsaName, Sramecc == on ? ":sramecc+" : ":sramecc-",
            MAX_ISA_NAME_SIZE - strlen(ExpectedIsaName));
  }

  if (Isa->XnackSupported && Xnack != any) {
    strncat(ExpectedIsaName, Xnack == on ? ":xnack+" : ":xnack-",
            MAX_ISA_NAME_SIZE - strlen(ExpectedIsaName));
  }

  return true;
}

void checkIsaName(amd_comgr_data_t Data, const char *InputIsaName,
                  const char *ExpectedIsaName) {
  size_t Size;
  char *IsaName = NULL;
  amd_comgr_status_t Status;

  Status = amd_comgr_get_data_isa_name(Data, &Size, IsaName);
  checkError(Status, "amd_comgr_get_data_isa_name");

  IsaName = malloc(Size);
  if (!IsaName) {
    printf("cannot allocate %zu bytes for isa_name\n", Size);
    exit(1);
  }

  Status = amd_comgr_get_data_isa_name(Data, &Size, IsaName);
  checkError(Status, "amd_comgr_get_data_isa_name");

  if (strcmp(IsaName, ExpectedIsaName)) {
    printf(
        "ISA name match failed: input '%s', expected '%s' but produced '%s'\n",
        InputIsaName, ExpectedIsaName, IsaName);
    exit(1);
  }

  free(IsaName);
}

void compileAndTestIsaName(const char *IsaName, const char *ExpectedIsaName,
                           const char *Options[], size_t OptionsCount) {
  char *BufSource;
  size_t SizeSource;
  amd_comgr_data_t DataSource, DataReloc, DataExec;
  amd_comgr_status_t Status;
  amd_comgr_data_set_t DataSetIn, DataSetBc, DataSetLinked, DataSetReloc,
      DataSetExec;
  amd_comgr_action_info_t DataAction;

  SizeSource = setBuf(TEST_OBJ_DIR "/shared.cl", &BufSource);

  Status = amd_comgr_create_data_set(&DataSetIn);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataSource);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataSource, SizeSource, BufSource);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataSource, "shared.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetIn, DataSource);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction, IsaName);
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status =
      amd_comgr_action_info_set_option_list(DataAction, Options, OptionsCount);
  checkError(Status, "amd_comgr_action_info_set_option_list");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetIn, DataSetBc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetLinked);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, DataAction,
                               DataSetBc, DataSetLinked);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               DataAction, DataSetLinked, DataSetReloc);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_get_data(
      DataSetReloc, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &DataReloc);
  checkError(Status, "amd_comgr_action_data_get_data");

  Status = amd_comgr_create_data_set(&DataSetExec);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetExec);
  checkError(Status, "amd_comgr_do_action");

  Status = amd_comgr_action_data_get_data(
      DataSetExec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &DataExec);
  checkError(Status, "amd_comgr_action_data_get_data");

  checkIsaName(DataReloc, IsaName, ExpectedIsaName);
  checkIsaName(DataExec, IsaName, ExpectedIsaName);
  printf("ISA name matched %s -> %s\n", IsaName, ExpectedIsaName);

  Status = amd_comgr_release_data(DataSource);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataReloc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataExec);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetIn);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetLinked);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetExec);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  free(BufSource);
}

void testIsaName(char *Name, const char *Features) {
  char IsaName[MAX_ISA_NAME_SIZE];
  char ExpectedIsaName[MAX_ISA_NAME_SIZE];

  strncpy(IsaName, Name, MAX_ISA_NAME_SIZE);
  strncat(IsaName, Features, MAX_ISA_NAME_SIZE - 1);

  const char *V4Options[] = {"-mcode-object-version=4"};
  size_t V4OptionsCount = sizeof(V4Options) / sizeof(V4Options[0]);

  const char *V6Options[] = {"-mcode-object-version=6"};
  size_t V6OptionsCount = sizeof(V6Options) / sizeof(V6Options[0]);

  // Test object code v6 so generic targets are available.
  bool NeedsCOV6;
  if (getExpectedIsaName(6, IsaName, ExpectedIsaName, &NeedsCOV6)) {
    if (NeedsCOV6) {
      printf("V6 : ");
      compileAndTestIsaName(IsaName, IsaName, V6Options, V6OptionsCount);
    } else {
      printf("V4 : ");
      compileAndTestIsaName(IsaName, IsaName, V4Options, V4OptionsCount);
    }
  }
}

int main(int argc, char *argv[]) {
  size_t IsaCount;
  amd_comgr_status_t Status;

  Status = amd_comgr_get_isa_count(&IsaCount);
  checkError(Status, "amd_comgr_get_isa_count");

  for (size_t I = 0; I < IsaCount; I++) {
    const char *Name;
    char IsaName[MAX_ISA_NAME_SIZE];

    Status = amd_comgr_get_isa_name(I, &Name);
    checkError(Status, "amd_comgr_get_isa_name");

    strncpy(IsaName, Name, MAX_ISA_NAME_SIZE);

    testIsaName(IsaName, "");

    for (size_t I = 0; I < IsaFeaturesSize; I++) {
      if (strncmp(IsaName, IsaFeatures[I].IsaName, MAX_ISA_NAME_SIZE) == 0) {

        if (IsaFeatures[I].SrameccSupported) {
          testIsaName(IsaName, ":sramecc+");
          testIsaName(IsaName, ":sramecc-");
        }

        if (IsaFeatures[I].XnackSupported) {
          testIsaName(IsaName, ":xnack+");
          testIsaName(IsaName, ":xnack-");
        }

        if (IsaFeatures[I].SrameccSupported && IsaFeatures[I].XnackSupported) {
          testIsaName(IsaName, ":sramecc+:xnack+");
          testIsaName(IsaName, ":sramecc+:xnack-");
          testIsaName(IsaName, ":sramecc-:xnack+");
          testIsaName(IsaName, ":sramecc-:xnack-");
        }

        break;
      }
    }
  }

  return 0;
}
