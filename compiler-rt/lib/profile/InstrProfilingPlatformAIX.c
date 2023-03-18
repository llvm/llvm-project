/*===- InstrProfilingPlatformAIX.c - Profile data AIX platform ------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#if defined(_AIX)

#include "InstrProfiling.h"

// Empty stubs to allow linking object files using the registration-based scheme
COMPILER_RT_VISIBILITY
void __llvm_profile_register_function(void *Data_) {}

COMPILER_RT_VISIBILITY
void __llvm_profile_register_names_function(void *NamesStart,
                                            uint64_t NamesSize) {}

// The __start_SECNAME and __stop_SECNAME symbols (for SECNAME \in
// {"__llvm_prf_cnts", "__llvm_prf_data", "__llvm_prf_name", "__llvm_prf_vnds"})
// are always live when linking on AIX, regardless if the .o's being linked
// reference symbols from the profile library (for example when no files were
// compiled with -fprofile-generate). That's because these symbols are kept
// alive through references in constructor functions that are always live in the
// default linking model on AIX (-bcdtors:all). The __start_SECNAME and
// __stop_SECNAME symbols are only resolved by the linker when the SECNAME
// section exists. So for the scenario where the user objects have no such
// section (i.e. when they are compiled with -fno-profile-generate), we always
// define these zero length variables in each of the above 4 sections.
static int dummy_cnts[0] COMPILER_RT_SECTION(
    COMPILER_RT_SEG INSTR_PROF_CNTS_SECT_NAME);
static int dummy_data[0] COMPILER_RT_SECTION(
    COMPILER_RT_SEG INSTR_PROF_DATA_SECT_NAME);
static const int dummy_name[0] COMPILER_RT_SECTION(
    COMPILER_RT_SEG INSTR_PROF_NAME_SECT_NAME);
static int dummy_vnds[0] COMPILER_RT_SECTION(
    COMPILER_RT_SEG INSTR_PROF_VNODES_SECT_NAME);

// To avoid GC'ing of the dummy variables by the linker, reference them in an
// array and reference the array in the runtime registration code
// (InstrProfilingRuntime.cpp)
COMPILER_RT_VISIBILITY
void *__llvm_profile_keep[] = {(void *)&dummy_cnts, (void *)&dummy_data,
                               (void *)&dummy_name, (void *)&dummy_vnds};

#endif
