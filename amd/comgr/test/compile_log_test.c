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

# undef unsetenv
# if !HAVE_DECL_UNSETENV
#  if VOID_UNSETENV
extern void unsetenv (const char *);
#  else
extern int unsetenv (const char *);
#  endif
# endif

int main(int argc, char *argv[]) {

  // For this test to pass when redirecting logs to stdout,
  // we need to temporarily undo the redirect
  if (getenv("AMD_COMGR_REDIRECT_LOGS") &&
      ( !strcmp("stdout", getenv("AMD_COMGR_REDIRECT_LOGS")) ||
        !strcmp("stderr", getenv("AMD_COMGR_REDIRECT_LOGS")) ))
      unsetenv("AMD_COMGR_REDIRECT_LOGS");

  amd_comgr_data_t DataCl, DataAsm, DataBc, DataReloc;
  amd_comgr_data_set_t DataSetOut, DataSetCl, DataSetAsm, DataSetBc,
      DataSetReloc;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  size_t Count;
  const char *Buf = "invalid";
  size_t Size = strlen(Buf);

  Status = amd_comgr_create_data_set(&DataSetCl);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataCl);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataCl, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataCl, "invalid.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetCl, DataCl);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data_set(&DataSetAsm);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataAsm);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataAsm, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataAsm, "invalid.s");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetAsm, DataAsm);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_BC, &DataBc);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataBc, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataBc, "invalid.bc");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetBc, DataBc);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_data_set(&DataSetReloc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &DataReloc);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataReloc, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataReloc, "invalid.o");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetReloc, DataReloc);
  checkError(Status, "amd_comgr_data_set_add");

  Status = amd_comgr_create_action_info(&DataAction);
  checkError(Status, "amd_comgr_create_action_info");
  Status = amd_comgr_action_info_set_language(DataAction,
                                              AMD_COMGR_LANGUAGE_OPENCL_1_2);
  checkError(Status, "amd_comgr_action_info_set_language");
  Status = amd_comgr_action_info_set_isa_name(DataAction,
                                              "amdgcn-amd-amdhsa--gfx900");
  checkError(Status, "amd_comgr_action_info_set_isa_name");
  Status = amd_comgr_action_info_set_logging(DataAction, true);
  checkError(Status, "amd_comgr_action_info_set_logging");

  // AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetCl, DataSetOut);
  checkLogs("COMPILE_SOURCE_TO_BC", DataSetOut,
            "error: unknown type name 'invalid'");
  checkLogs("COMPILE_SOURCE_TO_BC", DataSetOut, "2 errors generated.");

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC Failed: "
           "produced %zu LOG objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_LINK_BC_TO_BC

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_BC_TO_BC, DataAction,
                               DataSetBc, DataSetOut);
  checkLogs("LINK_BC_TO_BC", DataSetOut, "error: expected top-level entity");

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_BC_TO_BC Failed: "
           "produced %zu LOG objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                               DataAction, DataSetAsm, DataSetOut);
  checkLogs("ASSEMBLE_SOURCE_TO_RELOCATABLE", DataSetOut,
            "error: invalid instruction");

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                               DataAction, DataSetBc, DataSetOut);
  checkLogs("CODEGEN_BC_TO_RELOCATABLE", DataSetOut,
            "error: expected top-level entity");
  checkLogs("CODEGEN_BC_TO_RELOCATABLE", DataSetOut, "1 error generated.");

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  // AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE

  Status = amd_comgr_create_data_set(&DataSetOut);
  checkError(Status, "amd_comgr_create_data_set");

  Status = amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                               DataAction, DataSetReloc, DataSetOut);
  checkLogs("LINK_RELOCATABLE_TO_EXECUTABLE", DataSetOut, "unexpected EOF");

  Status =
      amd_comgr_action_data_count(DataSetOut, AMD_COMGR_DATA_KIND_LOG, &Count);
  checkError(Status, "amd_comgr_action_data_count");

  if (Count != 1) {
    printf("AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE Failed: "
           "produced %zu LOG objects (expected 1)\n",
           Count);
    exit(1);
  }

  Status = amd_comgr_destroy_data_set(DataSetOut);
  checkError(Status, "amd_comgr_destroy_data_set");

  Status = amd_comgr_release_data(DataCl);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataAsm);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataBc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_release_data(DataReloc);
  checkError(Status, "amd_comgr_release_data");
  Status = amd_comgr_destroy_data_set(DataSetCl);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetAsm);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetReloc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
}
