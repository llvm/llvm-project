//===- compile_log_remarks_test.c -----------------------------------------===//
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

#undef unsetenv
#ifdef _WIN32
#define unsetenv(name) _putenv_s(name, "")
#else
#if !HAVE_DECL_UNSETENV
#if VOID_UNSETENV
extern void unsetenv(const char *);
#else
extern int unsetenv(const char *);
#endif
#endif
#endif

int main(int argc, char *argv[]) {

  // For this test to pass when redirecting logs to stdout,
  // we need to temporarily undo the redirect
  if (getenv("AMD_COMGR_REDIRECT_LOGS") &&
      (!strcmp("stdout", getenv("AMD_COMGR_REDIRECT_LOGS")) ||
       !strcmp("stderr", getenv("AMD_COMGR_REDIRECT_LOGS"))))
    unsetenv("AMD_COMGR_REDIRECT_LOGS");

  amd_comgr_data_t DataCl;
  amd_comgr_data_set_t DataSetCl, DataSetBc, DataSetAsm;
  amd_comgr_action_info_t DataAction;
  amd_comgr_status_t Status;

  const char *Buf = "kernel void f() { volatile int x = 0; }";
  size_t Size = strlen(Buf);

  Status = amd_comgr_create_data_set(&DataSetCl);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &DataCl);
  checkError(Status, "amd_comgr_create_data");
  Status = amd_comgr_set_data(DataCl, Size, Buf);
  checkError(Status, "amd_comgr_set_data");
  Status = amd_comgr_set_data_name(DataCl, "empty.cl");
  checkError(Status, "amd_comgr_set_data_name");
  Status = amd_comgr_data_set_add(DataSetCl, DataCl);
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

  Status = amd_comgr_create_data_set(&DataSetBc);
  checkError(Status, "amd_comgr_create_data_set");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                               DataAction, DataSetCl, DataSetBc);
  checkError(Status, "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC");
  checkCount("AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC", DataSetBc,
             AMD_COMGR_DATA_KIND_BC, 1);

  Status = amd_comgr_create_data_set(&DataSetAsm);
  checkError(Status, "amd_comgr_create_data_set");
  const char *Options[] = {"-Rpass-analysis=prolog"};
  size_t Count = sizeof(Options) / sizeof(Options[0]);
  Status = amd_comgr_action_info_set_option_list(DataAction, Options, Count);
  checkError(Status, "amd_comgr_action_info_set_option_list");
  Status = amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                               DataAction, DataSetBc, DataSetAsm);
  checkError(Status, "AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY");
  checkCount("AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY", DataSetAsm,
             AMD_COMGR_DATA_KIND_SOURCE, 1);

  checkLogs("AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY", DataSetAsm,
            "remark: <unknown>:0:0: 8 stack bytes in function 'f' "
            "[-Rpass-analysis=prologepilog]");

  Status = amd_comgr_destroy_data_set(DataSetCl);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetBc);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_data_set(DataSetAsm);
  checkError(Status, "amd_comgr_destroy_data_set");
  Status = amd_comgr_destroy_action_info(DataAction);
  checkError(Status, "amd_comgr_destroy_action_info");
  Status = amd_comgr_release_data(DataCl);
  checkError(Status, "amd_comgr_release_data");
}
