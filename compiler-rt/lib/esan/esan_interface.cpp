//===-- esan_interface.cpp ------------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "esan_interface_internal.h"
#include "esan.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

using namespace __esan; // NOLINT

void __esan_init(ToolType Tool) {
  initializeLibrary(Tool);
}

void __esan_aligned_load1(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 1, false);
}

void __esan_aligned_load2(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 2, false);
}

void __esan_aligned_load4(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 4, false);
}

void __esan_aligned_load8(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 8, false);
}

void __esan_aligned_load16(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 16, false);
}

void __esan_aligned_store1(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 1, true);
}

void __esan_aligned_store2(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 2, true);
}

void __esan_aligned_store4(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 4, true);
}

void __esan_aligned_store8(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 8, true);
}

void __esan_aligned_store16(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 16, true);
}

void __esan_unaligned_load2(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 2, false);
}

void __esan_unaligned_load4(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 4, false);
}

void __esan_unaligned_load8(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 8, false);
}

void __esan_unaligned_load16(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 16, false);
}

void __esan_unaligned_store2(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 2, true);
}

void __esan_unaligned_store4(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 4, true);
}

void __esan_unaligned_store8(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 8, true);
}

void __esan_unaligned_store16(void *Addr) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, 16, true);
}

void __esan_unaligned_loadN(void *Addr, uptr Size) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, Size, false);
}

void __esan_unaligned_storeN(void *Addr, uptr Size) {
  processRangeAccess(GET_CALLER_PC(), (uptr)Addr, Size, true);
}
