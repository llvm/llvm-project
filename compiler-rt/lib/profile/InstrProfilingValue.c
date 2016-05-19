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

COMPILER_RT_VISIBILITY uint32_t VPMaxNumValsPerSite =
    INSTR_PROF_MAX_NUM_VAL_PER_SITE;

COMPILER_RT_VISIBILITY void lprofSetupValueProfiler() {
  const char *Str = 0;
  Str = getenv("LLVM_VP_MAX_NUM_VALS_PER_SITE");
  if (Str && Str[0])
    VPMaxNumValsPerSite = atoi(Str);
  if (VPMaxNumValsPerSite > INSTR_PROF_MAX_NUM_VAL_PER_SITE)
    VPMaxNumValsPerSite = INSTR_PROF_MAX_NUM_VAL_PER_SITE;
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
    if (TargetValue == CurrentVNode->Value) {
      CurrentVNode->Count++;
      return;
    }
    PrevVNode = CurrentVNode;
    CurrentVNode = CurrentVNode->Next;
    ++VDataCount;
  }

  if (VDataCount >= VPMaxNumValsPerSite)
    return;

  CurrentVNode = (ValueProfNode *)calloc(1, sizeof(ValueProfNode));
  if (!CurrentVNode)
    return;

  CurrentVNode->Value = TargetValue;
  CurrentVNode->Count++;

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

/*
 * A wrapper struct that represents value profile runtime data.
 * Like InstrProfRecord class which is used by profiling host tools,
 * ValueProfRuntimeRecord also implements the abstract intefaces defined in
 * ValueProfRecordClosure so that the runtime data can be serialized using
 * shared C implementation.
 */
typedef struct ValueProfRuntimeRecord {
  const __llvm_profile_data *Data;
  ValueProfNode **NodesKind[IPVK_Last + 1];
  uint8_t **SiteCountArray;
} ValueProfRuntimeRecord;

/* ValueProfRecordClosure Interface implementation. */

static uint32_t getNumValueSitesRT(const void *R, uint32_t VK) {
  return ((const ValueProfRuntimeRecord *)R)->Data->NumValueSites[VK];
}

static uint32_t getNumValueDataRT(const void *R, uint32_t VK) {
  uint32_t S = 0, I;
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  if (Record->SiteCountArray[VK] == INSTR_PROF_NULLPTR)
    return 0;
  for (I = 0; I < Record->Data->NumValueSites[VK]; I++)
    S += Record->SiteCountArray[VK][I];
  return S;
}

static uint32_t getNumValueDataForSiteRT(const void *R, uint32_t VK,
                                         uint32_t S) {
  const ValueProfRuntimeRecord *Record = (const ValueProfRuntimeRecord *)R;
  return Record->SiteCountArray[VK][S];
}

static ValueProfRuntimeRecord RTRecord;
static ValueProfRecordClosure RTRecordClosure = {
    &RTRecord,          INSTR_PROF_NULLPTR, /* GetNumValueKinds */
    getNumValueSitesRT, getNumValueDataRT,  getNumValueDataForSiteRT,
    INSTR_PROF_NULLPTR, /* RemapValueData */
    INSTR_PROF_NULLPTR, /* GetValueForSite, */
    INSTR_PROF_NULLPTR  /* AllocValueProfData */
};

static uint32_t
initializeValueProfRuntimeRecord(const __llvm_profile_data *Data,
                                 uint8_t *SiteCountArray[]) {
  unsigned I, J, S = 0, NumValueKinds = 0;
  ValueProfNode **Nodes = (ValueProfNode **)Data->Values;
  RTRecord.Data = Data;
  RTRecord.SiteCountArray = SiteCountArray;
  for (I = 0; I <= IPVK_Last; I++) {
    uint16_t N = Data->NumValueSites[I];
    if (!N)
      continue;

    NumValueKinds++;

    RTRecord.NodesKind[I] = Nodes ? &Nodes[S] : INSTR_PROF_NULLPTR;
    for (J = 0; J < N; J++) {
      /* Compute value count for each site. */
      uint32_t C = 0;
      ValueProfNode *Site =
          Nodes ? RTRecord.NodesKind[I][J] : INSTR_PROF_NULLPTR;
      while (Site) {
        C++;
        Site = Site->Next;
      }
      if (C > UCHAR_MAX)
        C = UCHAR_MAX;
      RTRecord.SiteCountArray[I][J] = C;
    }
    S += N;
  }
  return NumValueKinds;
}

static ValueProfNode *getNextNValueData(uint32_t VK, uint32_t Site,
                                        InstrProfValueData *Dst,
                                        ValueProfNode *StartNode, uint32_t N) {
  unsigned I;
  ValueProfNode *VNode = StartNode ? StartNode : RTRecord.NodesKind[VK][Site];
  for (I = 0; I < N; I++) {
    Dst[I].Value = VNode->Value;
    Dst[I].Count = VNode->Count;
    VNode = VNode->Next;
  }
  return VNode;
}

static uint32_t getValueProfDataSizeWrapper() {
  return getValueProfDataSize(&RTRecordClosure);
}

static uint32_t getNumValueDataForSiteWrapper(uint32_t VK, uint32_t S) {
  return getNumValueDataForSiteRT(&RTRecord, VK, S);
}

static VPDataReaderType TheVPDataReader = {
    initializeValueProfRuntimeRecord, getValueProfRecordHeaderSize,
    getFirstValueProfRecord,          getNumValueDataForSiteWrapper,
    getValueProfDataSizeWrapper,      getNextNValueData};

COMPILER_RT_VISIBILITY VPDataReaderType *lprofGetVPDataReader() {
  return &TheVPDataReader;
}
