/*===- InstrProfilingPlatformOther.c - Profile data default platform ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

#if !defined(__APPLE__) && !defined(__linux__) && !defined(__FreeBSD__)
#include <stdlib.h>

static const __llvm_profile_data *DataFirst = NULL;
static const __llvm_profile_data *DataLast = NULL;
static const char *NamesFirst = NULL;
static const char *NamesLast = NULL;
static uint64_t *CountersFirst = NULL;
static uint64_t *CountersLast = NULL;

/*!
 * \brief Register an instrumented function.
 *
 * Calls to this are emitted by clang with -fprofile-instr-generate.  Such
 * calls are only required (and only emitted) on targets where we haven't
 * implemented linker magic to find the bounds of the sections.
 */
COMPILER_RT_VISIBILITY
void __llvm_profile_register_function(void *Data_) {
  /* TODO: Only emit this function if we can't use linker magic. */
  const __llvm_profile_data *Data = (__llvm_profile_data *)Data_;
  if (!DataFirst) {
    DataFirst = Data;
    DataLast = Data + 1;
    NamesFirst = Data->NamePtr;
    NamesLast = (const char *)Data->NamePtr + Data->NameSize;
    CountersFirst = Data->CounterPtr;
    CountersLast = (uint64_t *)Data->CounterPtr + Data->NumCounters;
    return;
  }

#define UPDATE_FIRST(First, New) First = New < First ? New : First
  UPDATE_FIRST(DataFirst, Data);
  UPDATE_FIRST(NamesFirst, (const char *)Data->NamePtr);
  UPDATE_FIRST(CountersFirst, (uint64_t *)Data->CounterPtr);
#undef UPDATE_FIRST

#define UPDATE_LAST(Last, New) Last = New > Last ? New : Last
  UPDATE_LAST(DataLast, Data + 1);
  UPDATE_LAST(NamesLast, (const char *)Data->NamePtr + Data->NameSize);
  UPDATE_LAST(CountersLast, (uint64_t *)Data->CounterPtr + Data->NumCounters);
#undef UPDATE_LAST
}

COMPILER_RT_VISIBILITY
const __llvm_profile_data *__llvm_profile_begin_data(void) { return DataFirst; }
COMPILER_RT_VISIBILITY
const __llvm_profile_data *__llvm_profile_end_data(void) { return DataLast; }
COMPILER_RT_VISIBILITY
const char *__llvm_profile_begin_names(void) { return NamesFirst; }
COMPILER_RT_VISIBILITY
const char *__llvm_profile_end_names(void) { return NamesLast; }
COMPILER_RT_VISIBILITY
uint64_t *__llvm_profile_begin_counters(void) { return CountersFirst; }
COMPILER_RT_VISIBILITY
uint64_t *__llvm_profile_end_counters(void) { return CountersLast; }
#endif
