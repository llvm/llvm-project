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
    return NULL;                                                               \
  }

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

ValueProfData *allocValueProfDataRT(size_t TotalSizeInBytes) {
  return (ValueProfData *)calloc(TotalSizeInBytes, 1);
}

/*
 * The value profiler runtime library stores the value profile data
 * for a given function in \c NumValueSites and \c Nodes structures.
 * \c ValueProfRuntimeRecord class is used to encapsulate the runtime
 * profile data and provides fast interfaces to retrieve the profile
 * information. This interface is used to initialize the runtime record
 * and pre-compute the information needed for efficient implementation
 * of callbacks required by ValueProfRecordClosure class.
 */
static int
initializeValueProfRuntimeRecord(ValueProfRuntimeRecord *RuntimeRecord,
                                 const uint16_t *NumValueSites,
                                 ValueProfNode **Nodes) {
  unsigned I, S = 0, NumValueKinds = 0;
  RuntimeRecord->NumValueSites = NumValueSites;
  RuntimeRecord->Nodes = Nodes;
  for (I = 0; I <= IPVK_Last; I++) {
    uint16_t N = NumValueSites[I];
    if (!N)
      continue;
    NumValueKinds++;

    RuntimeRecord->NodesKind[I] = Nodes ? &Nodes[S] : INSTR_PROF_NULLPTR;
    S += N;
  }
  RuntimeRecord->NumValueKinds = NumValueKinds;
  return 0;
}

static void
finalizeValueProfRuntimeRecord(ValueProfRuntimeRecord *RuntimeRecord) {}

/* ValueProfRecordClosure Interface implementation for
 * ValueProfDataRuntimeRecord.  */
static uint32_t getNumValueKindsRT(const void *R) {
  return ((const ValueProfRuntimeRecord *)R)->NumValueKinds;
}

static uint32_t getNumValueSitesRT(const void *R, uint32_t VK) {
  return ((const ValueProfRuntimeRecord *)R)->NumValueSites[VK];
}

static uint32_t getNumValueDataForSiteRT(const void *R, uint32_t VK,
                                         uint32_t S) {
  uint32_t C = 0;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  ValueProfNode *Site =
      Record->NodesKind[VK] ? Record->NodesKind[VK][S] : INSTR_PROF_NULLPTR;
  while (Site) {
    C++;
    Site = Site->Next;
  }
  if (C > UCHAR_MAX)
    C = UCHAR_MAX;

  return C;
}

static uint32_t getNumValueDataRT(const void *R, uint32_t VK) {
  unsigned I, S = 0;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  for (I = 0; I < Record->NumValueSites[VK]; I++)
    S += getNumValueDataForSiteRT(Record, VK, I);
  return S;
}

static void getValueForSiteRT(const void *R, InstrProfValueData *Dst,
                              uint32_t VK, uint32_t S) {
  unsigned I, N = 0;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  N = getNumValueDataForSiteRT(R, VK, S);
  if (N == 0)
    return;
  ValueProfNode *VNode = Record->NodesKind[VK][S];
  for (I = 0; I < N; I++) {
    Dst[I] = VNode->VData;
    VNode = VNode->Next;
  }
}

static ValueProfRecordClosure RTRecordClosure = {
    INSTR_PROF_NULLPTR, getNumValueKindsRT,       getNumValueSitesRT,
    getNumValueDataRT,  getNumValueDataForSiteRT, INSTR_PROF_NULLPTR,
    getValueForSiteRT,  allocValueProfDataRT};

/*
 * Return a ValueProfData instance that stores the data collected
 * from runtime. If \c DstData is provided by the caller, the value
 * profile data will be store in *DstData and DstData is returned,
 * otherwise the method will allocate space for the value data and
 * return pointer to the newly allocated space.
 */
static ValueProfData *
serializeValueProfDataFromRT(const ValueProfRuntimeRecord *Record,
                             ValueProfData *DstData) {
  RTRecordClosure.Record = Record;
  return serializeValueProfDataFrom(&RTRecordClosure, DstData);
}

/*
 * Return the size of ValueProfData structure to store data
 * recorded in the runtime record.
 */
static uint32_t getValueProfDataSizeRT(const ValueProfRuntimeRecord *Record) {
  RTRecordClosure.Record = Record;
  return getValueProfDataSize(&RTRecordClosure);
}

COMPILER_RT_VISIBILITY struct ValueProfData *
lprofGatherValueProfData(const __llvm_profile_data *Data) {
  ValueProfData *VD = NULL;
  ValueProfRuntimeRecord R;
  if (initializeValueProfRuntimeRecord(&R, Data->NumValueSites, Data->Values))
    PROF_OOM_RETURN("Failed to write value profile data ");

  /* Compute the size of ValueProfData from this runtime record.  */
  if (getNumValueKindsRT(&R) != 0) {
    uint32_t VS = getValueProfDataSizeRT(&R);
    VD = (ValueProfData *)calloc(VS, sizeof(uint8_t));
    if (!VD)
      PROF_OOM_RETURN("Failed to write value profile data ");
    VD->TotalSize = VS;
    serializeValueProfDataFromRT(&R, VD);
  }
  finalizeValueProfRuntimeRecord(&R);

  return VD;
}
