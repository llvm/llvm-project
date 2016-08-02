/*===- InstrProfilingValue.c - Support library for PGO instrumentation ----===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define INSTR_PROF_VALUE_PROF_DATA
#define INSTR_PROF_COMMON_API_IMPL
#include "InstrProfData.inc"

#define PROF_OOM(Msg) PROF_ERR(Msg ":%s\n", "Out of memory");
#define PROF_OOM_RETURN(Msg)                                                   \
  {                                                                            \
    PROF_OOM(Msg)                                                              \
    free(ValueDataArray);                                                      \
    return NULL;                                                               \
  }

#if COMPILER_RT_HAS_ATOMICS != 1
COMPILER_RT_VISIBILITY
uint32_t BoolCmpXchg(void **Ptr, void *OldV, void *NewV) {
  void *R = *Ptr;
  if (R == OldV) {
    *Ptr = NewV;
    return 1;
  }
  return 0;
}
#endif

/* This method is only used in value profiler mock testing.  */
COMPILER_RT_VISIBILITY void
__llvm_profile_set_num_value_sites(__llvm_profile_data *Data,
                                   uint32_t ValueKind, uint16_t NumValueSites) {
  *((uint16_t *)&Data->NumValueSites[ValueKind]) = NumValueSites;
}

/* This method is only used in value profiler mock testing.  */
COMPILER_RT_VISIBILITY const __llvm_profile_data *
__llvm_profile_iterate_data(const __llvm_profile_data *Data) {
  return Data + 1;
}

/* This method is only used in value profiler mock testing.  */
COMPILER_RT_VISIBILITY void *
__llvm_get_function_addr(const __llvm_profile_data *Data) {
  return Data->FunctionPointer;
}

/* Allocate an array that holds the pointers to the linked lists of
 * value profile counter nodes. The number of element of the array
 * is the total number of value profile sites instrumented. Returns
 * 0 if allocation fails.
 */

static int allocateValueProfileCounters(__llvm_profile_data *Data) {
  uint64_t NumVSites = 0;
  uint32_t VKI;
  for (VKI = IPVK_First; VKI <= IPVK_Last; ++VKI)
    NumVSites += Data->NumValueSites[VKI];

  ValueProfNode **Mem =
      (ValueProfNode **)calloc(NumVSites, sizeof(ValueProfNode *));
  if (!Mem)
    return 0;
  if (!COMPILER_RT_BOOL_CMPXCHG(&Data->Values, 0, Mem)) {
    free(Mem);
    return 0;
  }
  return 1;
}

COMPILER_RT_VISIBILITY void
__llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                 uint32_t CounterIndex) {

  __llvm_profile_data *PData = (__llvm_profile_data *)Data;
  if (!PData)
    return;

  if (!PData->Values) {
    if (!allocateValueProfileCounters(PData))
      return;
  }

  ValueProfNode **ValueCounters = (ValueProfNode **)PData->Values;
  ValueProfNode *PrevVNode = NULL;
  ValueProfNode *CurrentVNode = ValueCounters[CounterIndex];

  uint8_t VDataCount = 0;
  while (CurrentVNode) {
    if (TargetValue == CurrentVNode->VData.Value) {
      CurrentVNode->VData.Count++;
      return;
    }
    PrevVNode = CurrentVNode;
    CurrentVNode = CurrentVNode->Next;
    ++VDataCount;
  }

  if (VDataCount >= INSTR_PROF_MAX_NUM_VAL_PER_SITE)
    return;

  CurrentVNode = (ValueProfNode *)calloc(1, sizeof(ValueProfNode));
  if (!CurrentVNode)
    return;

  CurrentVNode->VData.Value = TargetValue;
  CurrentVNode->VData.Count++;

  uint32_t Success = 0;
  if (!ValueCounters[CounterIndex])
    Success =
        COMPILER_RT_BOOL_CMPXCHG(&ValueCounters[CounterIndex], 0, CurrentVNode);
  else if (PrevVNode && !PrevVNode->Next)
    Success = COMPILER_RT_BOOL_CMPXCHG(&(PrevVNode->Next), 0, CurrentVNode);

  if (!Success) {
    free(CurrentVNode);
    return;
  }
}

COMPILER_RT_VISIBILITY ValueProfData **
__llvm_profile_gather_value_data(uint64_t *ValueDataSize) {
  size_t S = 0;
  __llvm_profile_data *I;
  ValueProfData **ValueDataArray;

  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();

  if (!ValueDataSize)
    return NULL;

  ValueDataArray =
      (ValueProfData **)calloc(DataEnd - DataBegin, sizeof(void *));
  if (!ValueDataArray)
    PROF_OOM_RETURN("Failed to write value profile data ");

  /*
   * Compute the total Size of the buffer to hold ValueProfData
   * structures for functions with value profile data.
   */
  for (I = (__llvm_profile_data *)DataBegin; I != DataEnd; ++I) {
    ValueProfRuntimeRecord R;
    if (initializeValueProfRuntimeRecord(&R, I->NumValueSites, I->Values))
      PROF_OOM_RETURN("Failed to write value profile data ");

    /* Compute the size of ValueProfData from this runtime record.  */
    if (getNumValueKindsRT(&R) != 0) {
      ValueProfData *VD = NULL;
      uint32_t VS = getValueProfDataSizeRT(&R);
      VD = (ValueProfData *)calloc(VS, sizeof(uint8_t));
      if (!VD)
        PROF_OOM_RETURN("Failed to write value profile data ");
      serializeValueProfDataFromRT(&R, VD);
      ValueDataArray[I - DataBegin] = VD;
      S += VS;
    }
    finalizeValueProfRuntimeRecord(&R);
  }

  if (!S) {
    free(ValueDataArray);
    ValueDataArray = NULL;
  }

  *ValueDataSize = S;
  return ValueDataArray;
}
