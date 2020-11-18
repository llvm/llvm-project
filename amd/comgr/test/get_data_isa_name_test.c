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
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ISA_NAME_SIZE 1024

typedef enum {none, off, on, any} mode_t;

typedef struct {
  const char *isa_name;
  bool supported_v2;
  bool sramecc_supported;
  mode_t sramecc_v2;
  bool xnack_supported;
  mode_t xnack_v2;
} isa_features_t;

/* Features supported based on https://llvm.org/docs/AMDGPUUsage.html . */
static isa_features_t isa_features[] = {
  //          ISA Name             V2         ------ SRAMECC ------  ------- XNACK -------
  //                               Supported  Supported  V2          Supported  V2
    {"amdgcn-amd-amdhsa--gfx700",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx701",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx702",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx703",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx704",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx801",  true,      false,     none,       true,      on},
    {"amdgcn-amd-amdhsa--gfx802",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx803",  true,      false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx810",  true,      false,     none,       true,      on},
    {"amdgcn-amd-amdhsa--gfx900",  true,      false,     none,       true,      any},
    {"amdgcn-amd-amdhsa--gfx902",  true,      false,     none,       true,      any},
    {"amdgcn-amd-amdhsa--gfx904",  true,      false,     none,       true,      any},
    {"amdgcn-amd-amdhsa--gfx906",  true,      true,      off,        true,      any},
    {"amdgcn-amd-amdhsa--gfx908",  false,     true,      none,       true,      none},
    {"amdgcn-amd-amdhsa--gfx909",  false,     false,     none,       true,      none},
    {"amdgcn-amd-amdhsa--gfx1010", false,     false,     none,       true,      none},
    {"amdgcn-amd-amdhsa--gfx1011", false,     false,     none,       true,      none},
    {"amdgcn-amd-amdhsa--gfx1012", false,     false,     none,       true,      none},
    {"amdgcn-amd-amdhsa--gfx1030", false,     false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx1031", false,     false,     none,       false,     none},
    {"amdgcn-amd-amdhsa--gfx1032", false,     false,     none,       false,     none}};

static int isa_features_size =
    sizeof(isa_features) / sizeof(isa_features[0]);

bool has_sub_string(const char *string, const char *sub) {
  return !strncmp(string, sub, strlen(sub));
}

bool get_expected_isa_name(unsigned code_object_version, const char *isa_name,
                           char *expected_isa_name) {
  char tokenized_isa_name[MAX_ISA_NAME_SIZE];

  strncpy(tokenized_isa_name, isa_name, MAX_ISA_NAME_SIZE);

  char *token = strtok(tokenized_isa_name, ":");
  isa_features_t *isa = NULL;
  for (unsigned i = 0; i < isa_features_size; i++) {
    if (strncmp(token, isa_features[i].isa_name, MAX_ISA_NAME_SIZE) == 0) {
      isa = &isa_features[i];
      break;
    }
  }
  if (!isa) {
    printf("The %s target is not supported by the test (update the isa_features table)\n", token);
    exit(1);
  }

  strncpy(expected_isa_name, isa->isa_name, MAX_ISA_NAME_SIZE);

  mode_t sramecc = any;
  mode_t xnack = any;

  token = strtok(NULL, ":");
  while (token != NULL) {
    if (strncmp(token, "sramecc", strlen("sramecc")) == 0 &&
        isa->sramecc_supported) {
      switch (token[strlen("sramecc")]) {
      case '-': sramecc = off; break;
      case '+': sramecc = on; break;
      }
    }

    if (strncmp(token, "xnack", strlen("xnack")) == 0 &&
        isa->xnack_supported) {
      switch (token[strlen("xnack")]) {
      case '-': xnack = off; break;
      case '+': xnack = on; break;
      }
    }

    token = strtok(NULL, ":");
  }

  switch (code_object_version) {
    case 2: {
      /* For a V2 ISA string which does not specify a feature, the code object
      * expected ISA string will have a supported feature set to ON. If the
      * feature setting does not match the default then it is not supported.
      */
      if (!isa->supported_v2)
        return false;
      if (isa->sramecc_supported) {
        if (sramecc == any) sramecc = on;
        if ((sramecc == on) != (isa->sramecc_v2 == on || isa->sramecc_v2 == any))
          return false;
      }
      if (isa->xnack_supported) {
        if (xnack == any) xnack = on;
        if ((xnack == on) != (isa->xnack_v2 == on || isa->xnack_v2 == any))
          return false;
      }
      break;
    }

    case 3: {
      /* If a supported feature is not specified in the ISA string then it will
       * be enabled in the expected isa.
       */
      if (isa->sramecc_supported) {
        if (sramecc == any) sramecc = on;
      }
      if (isa->xnack_supported) {
        if (xnack == any) xnack = on;
      }
      break;
    }

    case 4: {
      // All ISA strings are valid.
      return true;
    }

    default:
      printf("Code object V%u is not supported by the test (update the get_expected_isa_name)\n", code_object_version);
      exit(1);

  }

  strncpy(expected_isa_name, isa->isa_name, MAX_ISA_NAME_SIZE);

  if (isa->sramecc_supported && sramecc != any) {
    strncat(expected_isa_name,
            sramecc == on ? ":sramecc+" : ":sramecc-",
            MAX_ISA_NAME_SIZE - strlen(expected_isa_name));
  }

  if (isa->xnack_supported && xnack != any) {
    strncat(expected_isa_name,
            xnack == on ? ":xnack+" : ":xnack-",
            MAX_ISA_NAME_SIZE - strlen(expected_isa_name));
  }

  return true;
}

void check_isa_name(amd_comgr_data_t data, const char *input_isa_name,
                    const char *expected_isa_name) {
  size_t expected_size = strlen(expected_isa_name) + 1;

  size_t size;
  char *isa_name = NULL;
  amd_comgr_status_t status;

  status = amd_comgr_get_data_isa_name(data, &size, isa_name);
  checkError(status, "amd_comgr_get_data_isa_name");

  isa_name = malloc(size);
  if (!isa_name) {
    printf("cannot allocate %zu bytes for isa_name\n", size);
    exit(1);
  }

  status = amd_comgr_get_data_isa_name(data, &size, isa_name);
  checkError(status, "amd_comgr_get_data_isa_name");

  if (strcmp(isa_name, expected_isa_name)) {
    printf(
        "ISA name match failed: input '%s', expected '%s' but produced '%s'\n",
        input_isa_name, expected_isa_name, isa_name);
    exit(1);
  }

  free(isa_name);
}

void compile_and_test_isa_name(const char *isa_name,
                               const char *expected_isa_name,
                               const char *options[], size_t options_count) {
  char *buf_source;
  size_t size_source;
  amd_comgr_data_t data_source, data_reloc, data_exec;
  amd_comgr_status_t status;
  amd_comgr_data_set_t data_set_in, data_set_bc, data_set_linked,
      data_set_reloc, data_set_exec;
  amd_comgr_action_info_t data_action;

  size_source = setBuf(TEST_OBJ_DIR "/shared.cl", &buf_source);

  status = amd_comgr_create_data_set(&data_set_in);
  checkError(status, "amd_comgr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &data_source);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(data_source, size_source, buf_source);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(data_source, "shared.cl");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(data_set_in, data_source);
  checkError(status, "amd_comgr_data_set_add");

  status = amd_comgr_create_action_info(&data_action);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_language(data_action,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(status, "amd_comgr_action_info_set_language");
  status = amd_comgr_action_info_set_isa_name(data_action, isa_name);
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_option_list(data_action, options,
                                                 options_count);
  checkError(status, "amd_comgr_action_info_set_option_list");

  status = amd_comgr_create_data_set(&data_set_bc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               data_action, data_set_in, data_set_bc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&data_set_linked);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, data_action,
                               data_set_bc, data_set_linked);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_create_data_set(&data_set_reloc);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               data_action, data_set_linked, data_set_reloc);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_get_data(
      data_set_reloc, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &data_reloc);
  checkError(status, "amd_comgr_action_data_get_data");

  status = amd_comgr_create_data_set(&data_set_exec);
  checkError(status, "amd_comgr_create_data_set");
  status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               data_action, data_set_reloc, data_set_exec);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_action_data_get_data(
      data_set_exec, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &data_exec);
  checkError(status, "amd_comgr_action_data_get_data");

  check_isa_name(data_reloc, isa_name, expected_isa_name);
  check_isa_name(data_exec, isa_name, expected_isa_name);
  printf("ISA name matched %s -> %s\n", isa_name, expected_isa_name);

  status = amd_comgr_release_data(data_source);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(data_reloc);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_release_data(data_exec);
  checkError(status, "amd_comgr_release_data");
  status = amd_comgr_destroy_data_set(data_set_in);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(data_set_bc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(data_set_linked);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(data_set_reloc);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_data_set(data_set_exec);
  checkError(status, "amd_comgr_destroy_data_set");
  status = amd_comgr_destroy_action_info(data_action);
  checkError(status, "amd_comgr_destroy_action_info");
  free(buf_source);
}

void test_isa_name(char *name, const char *features) {
  char isa_name[MAX_ISA_NAME_SIZE];
  char expected_isa_name[MAX_ISA_NAME_SIZE];

  strncpy(isa_name, name, MAX_ISA_NAME_SIZE);
  strncat(isa_name, features, MAX_ISA_NAME_SIZE);

  const char *v2_options[] = {"-mllvm", "--amdhsa-code-object-version=2"};
  size_t v2_options_count = sizeof(v2_options) / sizeof(v2_options[0]);
  const char *v3_options[] = {"-mllvm", "--amdhsa-code-object-version=3"};
  size_t v3_options_count = sizeof(v3_options) / sizeof(v3_options[0]);
  const char *v4_options[] = {"-mllvm", "--amdhsa-code-object-version=4"};
  size_t v4_options_count = sizeof(v4_options) / sizeof(v4_options[0]);

  // Test object code v2.
  if (get_expected_isa_name(2, isa_name, expected_isa_name)) {
    printf("V2 : ");
    compile_and_test_isa_name(isa_name, expected_isa_name, v2_options,
                              v2_options_count);
  }

  // Test object code v3.
  if (get_expected_isa_name(3, isa_name, expected_isa_name)) {
    printf("V3 : ");
    compile_and_test_isa_name(isa_name, expected_isa_name, v3_options,
                              v3_options_count);
  }

  // Test object code v4.
  if (get_expected_isa_name(4, isa_name, expected_isa_name)) {
    printf("V4 : ");
    compile_and_test_isa_name(isa_name, isa_name, v4_options, v4_options_count);
  }
}

int main(int argc, char *argv[]) {
  size_t isa_count;
  amd_comgr_status_t status;

  status = amd_comgr_get_isa_count(&isa_count);
  checkError(status, "amd_comgr_get_isa_count");

  for (size_t i = 0; i < isa_count; i++) {
    const char *name;
    char isa_name[MAX_ISA_NAME_SIZE];

    status = amd_comgr_get_isa_name(i, &name);
    checkError(status, "amd_comgr_get_isa_name");

    strncpy(isa_name, name, MAX_ISA_NAME_SIZE);

    test_isa_name(isa_name, "");

    for (size_t i = 0; i < isa_features_size; i++)
      if (strncmp(isa_name, isa_features[i].isa_name, MAX_ISA_NAME_SIZE) == 0) {

        if (isa_features[i].sramecc_supported) {
          test_isa_name(isa_name, ":sramecc+");
          test_isa_name(isa_name, ":sramecc-");
        }

        if (isa_features[i].xnack_supported) {
          test_isa_name(isa_name, ":xnack+");
          test_isa_name(isa_name, ":xnack-");
        }

        if (isa_features[i].sramecc_supported &&
            isa_features[i].xnack_supported) {
          test_isa_name(isa_name, ":sramecc+:xnack+");
          test_isa_name(isa_name, ":sramecc+:xnack-");
          test_isa_name(isa_name, ":sramecc-:xnack+");
          test_isa_name(isa_name, ":sramecc-:xnack-");
        }

        break;
      }
  }

  return 0;
}
