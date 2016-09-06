/*===- InstrProfilingMergeFile.c - Profile in-process Merging  ------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
|*===----------------------------------------------------------------------===
|* This file defines APIs needed to support in-process merging for profile data
|* stored in files.
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingUtil.h"

#define INSTR_PROF_VALUE_PROF_DATA
#include "InstrProfData.inc"

void (*VPMergeHook)(ValueProfData *,
                    __llvm_profile_data *) = &lprofMergeValueProfData;

/* Merge value profile data pointed to by SrcValueProfData into
 * in-memory profile counters pointed by to DstData.  */
void lprofMergeValueProfData(ValueProfData *SrcValueProfData,
                             __llvm_profile_data *DstData) {
  unsigned I, S, V, C;
  InstrProfValueData *VData;
  ValueProfRecord *VR = getFirstValueProfRecord(SrcValueProfData);
  for (I = 0; I < SrcValueProfData->NumValueKinds; I++) {
    VData = getValueProfRecordValueData(VR);
    for (S = 0; S < VR->NumValueSites; S++) {
      uint8_t NV = VR->SiteCountArray[S];
      for (V = 0; V < NV; V++) {
        for (C = 0; C < VData[V].Count; C++)
          __llvm_profile_instrument_target(VData[V].Value, DstData, S);
      }
    }
    VR = getValueProfRecordNext(VR);
  }
}
