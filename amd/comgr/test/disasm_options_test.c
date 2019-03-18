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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

const char *expected_out = \
"\n"
":\tfile format ELF64-amdgpu\n"
"\n"
"Disassembly of section .text:\n"
"foo:\n"
"\ts_load_dwordx2 s[0:1], s[4:5], 0x0                         // 000000000100: C0060002 00000000 \n"
"\tv_mov_b32_e32 v2, 42                                       // 000000000108: 7E0402AA \n"
"\ts_waitcnt lgkmcnt(0)                                       // 00000000010C: BF8C007F \n"
"\tv_mov_b32_e32 v0, s0                                       // 000000000110: 7E000200 \n"
"\tv_mov_b32_e32 v1, s1                                       // 000000000114: 7E020201 \n"
"\tflat_store_dword v[0:1], v2                                // 000000000118: DC700000 00000200 \n"
"\ts_endpgm                                                   // 000000000120: BF810000 \n";

const char *expected_log = \
"amd_comgr_do_action:\n"
"\taction_kind: AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE\n"
"\tisa_name: amdgcn-amd-amdhsa--gfx803\n"
"\taction_options: -file-headers -invalid-option\n"
"\taction_path: \n"
"\tlanguage: AMD_COMGR_LANGUAGE_NONE\n"
": Unknown command line argument '-invalid-option'.  Try: ' -help'\n"
": Did you mean '-print-all-options'?\n";

void print_chars(const char *bytes, size_t count) {
  for (size_t i = 0; i < count; i++)
    printf("%c", bytes[i]);
}

void expect(const char *expected, const char *actual, size_t count) {
  if (strlen(expected) != count || strncmp(expected, actual, count)) {
    printf("FAILED: unexpected output\n");
    printf("expected:\n");
    printf("%s", expected);
    printf("actual:\n");
    print_chars(actual, count);
    exit(1);
  }
}

int main(int argc, char *argv[]) {
  size_t size;
  char *buf;
  char *bytes;
  amd_comgr_data_t dataIn, dataOut;
  amd_comgr_data_set_t dataSetIn, dataSetOut;
  amd_comgr_action_info_t dataAction;
  amd_comgr_status_t status;

  // Read input file
  size = setBuf(TEST_OBJ_DIR "/reloc1.o", &buf);

  status = amd_comgr_create_data_set(&dataSetIn);
  checkError(status, "amd_cogmr_create_data_set");

  status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &dataIn);
  checkError(status, "amd_comgr_create_data");
  status = amd_comgr_set_data(dataIn, size, buf);
  checkError(status, "amd_comgr_set_data");
  status = amd_comgr_set_data_name(dataIn, "DO_IN");
  checkError(status, "amd_comgr_set_data_name");
  status = amd_comgr_data_set_add(dataSetIn, dataIn);
  checkError(status, "amd_cogmr_data_set_add");

  status = amd_comgr_create_data_set(&dataSetOut);
  checkError(status, "amd_cogmr_create_data_set");

  status = amd_comgr_create_action_info(&dataAction);
  checkError(status, "amd_comgr_create_action_info");
  status = amd_comgr_action_info_set_isa_name(dataAction,
                                              "amdgcn-amd-amdhsa--gfx803");
  checkError(status, "amd_comgr_action_info_set_isa_name");
  status = amd_comgr_action_info_set_logging(dataAction, true);
  checkError(status, "amd_comgr_action_info_set_logging");
  status = amd_comgr_action_info_set_options(dataAction, "-file-headers -invalid-option");
  checkError(status, "amd_comgr_action_info_set_options");

  status =
      amd_comgr_do_action(AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE,
                          dataAction, dataSetIn, dataSetOut);
  checkError(status, "amd_comgr_do_action");

  status = amd_comgr_destroy_data_set(dataSetIn);
  checkError(status, "amd_comgr_destroy_data_set");

  size_t count;
  status = amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_SOURCE,
                                       &count);
  checkError(status, "amd_comgr_action_data_count");
  if (count != 1) {
    printf("wrong number of source data objects (%ld returned, expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_action_data_count(dataSetOut, AMD_COMGR_DATA_KIND_LOG,
                                       &count);
  checkError(status, "amd_comgr_action_data_count");
  if (count != 1) {
    printf("wrong number of log data objects (%ld returned, expected 1)\n",
           count);
    exit(1);
  }

  status = amd_comgr_action_data_get_data(
      dataSetOut, AMD_COMGR_DATA_KIND_SOURCE, 0, &dataOut);
  checkError(status, "amd_comgr_action_data_get_data");
  status = amd_comgr_get_data(dataOut, &count, NULL);
  checkError(status, "amd_comgr_get_data");
  bytes = (char *)calloc(count, sizeof(char));
  status = amd_comgr_get_data(dataOut, &count, bytes);
  checkError(status, "amd_comgr_get_data");
  expect(expected_out, bytes, count);
  free(bytes);
  status = amd_comgr_release_data(dataOut);
  checkError(status, "amd_comgr_release_data");

  status = amd_comgr_action_data_get_data(
      dataSetOut, AMD_COMGR_DATA_KIND_LOG, 0, &dataOut);
  checkError(status, "amd_comgr_action_data_get_data");
  status = amd_comgr_get_data(dataOut, &count, NULL);
  checkError(status, "amd_comgr_get_data");
  bytes = (char *)calloc(count, sizeof(char));
  status = amd_comgr_get_data(dataOut, &count, bytes);
  checkError(status, "amd_comgr_get_data");
  expect(expected_log, bytes, count);
  free(bytes);
  status = amd_comgr_release_data(dataOut);
  checkError(status, "amd_comgr_release_data");

  status = amd_comgr_destroy_data_set(dataSetOut);
  checkError(status, "amd_comgr_destroy_data_set");

  status = amd_comgr_destroy_action_info(dataAction);
  checkError(status, "amd_comgr_destroy_action_info");
  status = amd_comgr_release_data(dataIn);
  checkError(status, "amd_comgr_release_data");
  free(buf);

  return 0;
}
