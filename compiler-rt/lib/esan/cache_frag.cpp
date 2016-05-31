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

namespace __esan {

// This should be kept consistent with LLVM's EfficiencySanitizer StructInfo.
struct StructInfo {
  const char *StructName;
  u32 NumOfFields;
  u64 *FieldCounters;
  const char **FieldTypeNames;
};

// This should be kept consistent with LLVM's EfficiencySanitizer CacheFragInfo.
// The tool-specific information per compilation unit (module).
struct CacheFragInfo {
  const char *UnitName;
  u32 NumOfStructs;
  StructInfo *Structs;
};

//===-- Init/exit functions -----------------------------------------------===//

void processCacheFragCompilationUnitInit(void *Ptr) {
  CacheFragInfo *CacheFrag = (CacheFragInfo *)Ptr;
  VPrintf(2, "in esan::%s: %s with %u class(es)/struct(s)\n",
          __FUNCTION__, CacheFrag->UnitName, CacheFrag->NumOfStructs);
}

void processCacheFragCompilationUnitExit(void *Ptr) {
  CacheFragInfo *CacheFrag = (CacheFragInfo *)Ptr;
  VPrintf(2, "in esan::%s: %s with %u class(es)/struct(s)\n",
          __FUNCTION__, CacheFrag->UnitName, CacheFrag->NumOfStructs);
}

void initializeCacheFrag() {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
}

int finalizeCacheFrag() {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
  // FIXME: add the cache fragmentation final report.
  Report("%s is not finished: nothing yet to report\n", SanitizerToolName);
  return 0;
}

} // namespace __esan
