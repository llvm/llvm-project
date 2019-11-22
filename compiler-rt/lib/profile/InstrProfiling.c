/*===- InstrProfiling.c - Support library for PGO instrumentation ---------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"

#define INSTR_PROF_VALUE_PROF_DATA
#include "InstrProfData.inc"


COMPILER_RT_WEAK uint64_t INSTR_PROF_RAW_VERSION_VAR = INSTR_PROF_RAW_VERSION;

COMPILER_RT_VISIBILITY uint64_t __llvm_profile_get_magic(void) {
  return sizeof(void *) == sizeof(uint64_t) ? (INSTR_PROF_RAW_MAGIC_64)
                                            : (INSTR_PROF_RAW_MAGIC_32);
}

static unsigned ProfileDumped = 0;

COMPILER_RT_VISIBILITY unsigned lprofProfileDumped() {
  return ProfileDumped;
}

COMPILER_RT_VISIBILITY void lprofSetProfileDumped() {
  ProfileDumped = 1;
}

COMPILER_RT_VISIBILITY void __llvm_profile_set_dumped() {
  lprofSetProfileDumped();
}

/* Return the number of bytes needed to add to SizeInBytes to make it
 *   the result a multiple of 8.
 */
COMPILER_RT_VISIBILITY uint8_t
__llvm_profile_get_num_padding_bytes(uint64_t SizeInBytes) {
  return 7 & (sizeof(uint64_t) - SizeInBytes % sizeof(uint64_t));
}

COMPILER_RT_VISIBILITY uint64_t __llvm_profile_get_version(void) {
  return __llvm_profile_raw_version;
}

COMPILER_RT_VISIBILITY void __llvm_profile_reset_counters(void) {
  uint64_t *I = __llvm_profile_begin_counters();
  uint64_t *E = __llvm_profile_end_counters();

  memset(I, 0, sizeof(uint64_t) * (E - I));

  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DI;
  for (DI = DataBegin; DI < DataEnd; ++DI) {
    uint64_t CurrentVSiteCount = 0;
    uint32_t VKI, i;
    if (!DI->Values)
      continue;

    ValueProfNode **ValueCounters = (ValueProfNode **)DI->Values;

    for (VKI = IPVK_First; VKI <= IPVK_Last; ++VKI)
      CurrentVSiteCount += DI->NumValueSites[VKI];

    for (i = 0; i < CurrentVSiteCount; ++i) {
      ValueProfNode *CurrentVNode = ValueCounters[i];

      while (CurrentVNode) {
        CurrentVNode->Count = 0;
        CurrentVNode = CurrentVNode->Next;
      }
    }
  }
  ProfileDumped = 0;
}
