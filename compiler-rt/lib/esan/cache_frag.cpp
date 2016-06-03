//===-- cache_frag.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// This file contains cache fragmentation-specific code.
//===----------------------------------------------------------------------===//

#include "esan.h"
#include "sanitizer_common/sanitizer_addrhashmap.h"
#include "sanitizer_common/sanitizer_placement_new.h"

namespace __esan {

//===-- Struct field access counter runtime -------------------------------===//

// This should be kept consistent with LLVM's EfficiencySanitizer StructInfo.
struct StructInfo {
  const char *StructName;
  u32 NumFields;
  u64 *FieldCounters;
  const char **FieldTypeNames;
};

// This should be kept consistent with LLVM's EfficiencySanitizer CacheFragInfo.
// The tool-specific information per compilation unit (module).
struct CacheFragInfo {
  const char *UnitName;
  u32 NumStructs;
  StructInfo *Structs;
};

struct StructCounter {
  StructInfo *Struct;
  u64 Count;    // The total access count of the struct.
  u32 Variance; // Variance score for the struct layout access.
};

// We use StructHashMap to keep track of an unique copy of StructCounter.
typedef AddrHashMap<StructCounter, 31051> StructHashMap;
struct Context {
  StructHashMap StructMap;
  u32 NumStructs;
  u64 TotalCount; // The total access count of all structs.
};
static Context *Ctx;

static void registerStructInfo(CacheFragInfo *CacheFrag) {
  for (u32 i = 0; i < CacheFrag->NumStructs; ++i) {
    StructInfo *Struct = &CacheFrag->Structs[i];
    StructHashMap::Handle H(&Ctx->StructMap, (uptr)Struct->FieldCounters);
    if (H.created()) {
      VPrintf(2, " Register %s: %u fields\n",
              Struct->StructName, Struct->NumFields);
      H->Struct = Struct;
      ++Ctx->NumStructs;
    } else {
      VPrintf(2, " Duplicated %s: %u fields\n",
              Struct->StructName, Struct->NumFields);
    }
  }
}

static void unregisterStructInfo(CacheFragInfo *CacheFrag) {
  // FIXME: if the library is unloaded before finalizeCacheFrag, we should
  // collect the result for later report.
  for (u32 i = 0; i < CacheFrag->NumStructs; ++i) {
    StructInfo *Struct = &CacheFrag->Structs[i];
    StructHashMap::Handle H(&Ctx->StructMap, (uptr)Struct->FieldCounters, true);
    if (H.exists()) {
      VPrintf(2, " Unregister %s: %u fields\n",
              Struct->StructName, Struct->NumFields);
      --Ctx->NumStructs;
    } else {
      VPrintf(2, " Duplicated %s: %u fields\n",
              Struct->StructName, Struct->NumFields);
    }
  }
}

static void reportStructSummary() {
  // FIXME: iterate StructHashMap and generate the final report.
  Report("%s is not finished: nothing yet to report\n", SanitizerToolName);
}

//===-- Init/exit functions -----------------------------------------------===//

void processCacheFragCompilationUnitInit(void *Ptr) {
  CacheFragInfo *CacheFrag = (CacheFragInfo *)Ptr;
  VPrintf(2, "in esan::%s: %s with %u class(es)/struct(s)\n",
          __FUNCTION__, CacheFrag->UnitName, CacheFrag->NumStructs);
  registerStructInfo(CacheFrag);
}

void processCacheFragCompilationUnitExit(void *Ptr) {
  CacheFragInfo *CacheFrag = (CacheFragInfo *)Ptr;
  VPrintf(2, "in esan::%s: %s with %u class(es)/struct(s)\n",
          __FUNCTION__, CacheFrag->UnitName, CacheFrag->NumStructs);
  unregisterStructInfo(CacheFrag);
}

void initializeCacheFrag() {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
  // We use placement new to initialize Ctx before C++ static initializaion.
  // We make CtxMem 8-byte aligned for atomic operations in AddrHashMap.
  static u64 CtxMem[sizeof(Context) / sizeof(u64) + 1];
  Ctx = new(CtxMem) Context();
  Ctx->NumStructs = 0;
}

int finalizeCacheFrag() {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
  reportStructSummary();
  return 0;
}

} // namespace __esan
