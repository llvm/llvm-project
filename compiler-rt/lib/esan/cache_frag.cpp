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

//===-- Init/exit functions -----------------------------------------------===//

void processCacheFragCompilationUnitInit(void *Ptr) {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
}

void processCacheFragCompilationUnitExit(void *Ptr) {
  VPrintf(2, "in esan::%s\n", __FUNCTION__);
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
