//===- InstrProfilingRuntime.cpp - PGO runtime initialization -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" {

#include "InstrProfiling.h"

static int RegisterRuntime() {
  __llvm_profile_initialize();
  return 0;
}

#ifndef _AIX
/* int __llvm_profile_runtime  */
COMPILER_RT_VISIBILITY int INSTR_PROF_PROFILE_RUNTIME_VAR;

static int Registration = RegisterRuntime();
#else
extern COMPILER_RT_VISIBILITY void *__llvm_profile_keep[];
/* On AIX, when linking with -bcdtors:csect, the variable whose constructor does
 * the registration needs to be explicitly kept, hence we reuse the runtime hook
 * variable to do the registration since it'll be kept via the -u linker flag.
 * Create a volatile reference to __llvm_profile_keep to keep the array alive.*/
COMPILER_RT_VISIBILITY int INSTR_PROF_PROFILE_RUNTIME_VAR =
    ((void)*(void *volatile *)__llvm_profile_keep, RegisterRuntime());
#endif
}
