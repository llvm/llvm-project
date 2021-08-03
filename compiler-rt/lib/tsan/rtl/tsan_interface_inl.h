//===-- tsan_interface_inl.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_ptrauth.h"
#include "tsan_interface.h"
#include "tsan_rtl.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;

void __tsan_read1(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 1, kAccessRead);
}

void __tsan_read2(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, kAccessRead);
}

void __tsan_read4(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, kAccessRead);
}

void __tsan_read8(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, kAccessRead);
}

void __tsan_write1(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 1, kAccessWrite);
}

void __tsan_write2(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, kAccessWrite);
}

void __tsan_write4(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, kAccessWrite);
}

void __tsan_write8(void *addr) {
  MemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, kAccessWrite);
}

void __tsan_read1_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 1, kAccessRead);
}

void __tsan_read2_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 2, kAccessRead);
}

void __tsan_read4_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 4, kAccessRead);
}

void __tsan_read8_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 8, kAccessRead);
}

void __tsan_write1_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 1, kAccessWrite);
}

void __tsan_write2_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 2, kAccessWrite);
}

void __tsan_write4_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 4, kAccessWrite);
}

void __tsan_write8_pc(void *addr, void *pc) {
  MemoryAccess(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, 8, kAccessWrite);
}

void __tsan_vptr_update(void **vptr_p, void *new_val) {
  CHECK_EQ(sizeof(vptr_p), 8);
  if (*vptr_p != new_val) {
    ThreadState *thr = cur_thread();
    thr->is_vptr_access = true;
    MemoryAccess(thr, CALLERPC, (uptr)vptr_p, 8, kAccessWrite);
    thr->is_vptr_access = false;
  }
}

void __tsan_vptr_read(void **vptr_p) {
  CHECK_EQ(sizeof(vptr_p), 8);
  ThreadState *thr = cur_thread();
  thr->is_vptr_access = true;
  MemoryAccess(thr, CALLERPC, (uptr)vptr_p, 8, kAccessRead);
  thr->is_vptr_access = false;
}

void __tsan_func_entry(void *pc) { FuncEntry(cur_thread(), STRIP_PAC_PC(pc)); }

void __tsan_func_exit() { FuncExit(cur_thread()); }

void __tsan_ignore_thread_begin() { ThreadIgnoreBegin(cur_thread(), CALLERPC); }

void __tsan_ignore_thread_end() { ThreadIgnoreEnd(cur_thread()); }

void __tsan_read_range(void *addr, uptr size) {
  MemoryAccessRange(cur_thread(), CALLERPC, (uptr)addr, size, false);
}

void __tsan_write_range(void *addr, uptr size) {
  MemoryAccessRange(cur_thread(), CALLERPC, (uptr)addr, size, true);
}

void __tsan_read_range_pc(void *addr, uptr size, void *pc) {
  MemoryAccessRange(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, size, false);
}

void __tsan_write_range_pc(void *addr, uptr size, void *pc) {
  MemoryAccessRange(cur_thread(), STRIP_PAC_PC(pc), (uptr)addr, size, true);
}
