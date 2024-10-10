//===-- tsan_interface.cpp ------------------------------------------------===//
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

#include "tsan_interface.h"

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_ptrauth.h"
#include "tsan_interface_ann.h"
#include "tsan_rtl.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;

void __tsan_init() { Initialize(cur_thread_init()); }

void __tsan_flush_memory() {
  FlushShadowMemory();
}

void __tsan_read16_pc(void *addr, void *pc) {
  uptr pc_no_pac = STRIP_PAC_PC(pc);
  ThreadState *thr = cur_thread();
  MemoryAccess(thr, pc_no_pac, (uptr)addr, 8, kAccessRead);
  MemoryAccess(thr, pc_no_pac, (uptr)addr + 8, 8, kAccessRead);
}

void __tsan_write16_pc(void *addr, void *pc) {
  uptr pc_no_pac = STRIP_PAC_PC(pc);
  ThreadState *thr = cur_thread();
  MemoryAccess(thr, pc_no_pac, (uptr)addr, 8, kAccessWrite);
  MemoryAccess(thr, pc_no_pac, (uptr)addr + 8, 8, kAccessWrite);
}

// __tsan_unaligned_read/write calls are emitted by compiler.

template <unsigned int N>
void __tsan_unaligned_readx(const void *addr) {
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (unsigned int i = 0; i < N / 8; i++)
    UnalignedMemoryAccess(thr, pc, (uptr)addr + (i * 8), 8, kAccessRead);
}

template <unsigned int N>
void __tsan_unaligned_writex(void *addr) {
  uptr pc = CALLERPC;
  ThreadState *thr = cur_thread();
  for (unsigned int i = 0; i < N / 8; i++)
    UnalignedMemoryAccess(thr, pc, (uptr)addr + (i * 8), 8, kAccessWrite);
}

void __tsan_unaligned_read16(const void *addr) {
  __tsan_unaligned_readx<16>(addr);
}

void __tsan_unaligned_write16(void *addr) { __tsan_unaligned_writex<16>(addr); }

extern "C" void __tsan_unaligned_read32(const void *addr) {
  __tsan_unaligned_readx<32>(addr);
}

extern "C" void __tsan_unaligned_write32(void *addr) {
  __tsan_unaligned_writex<32>(addr);
}

extern "C" void __tsan_unaligned_read64(const void *addr) {
  __tsan_unaligned_readx<64>(addr);
}

extern "C" void __tsan_unaligned_write64(void *addr) {
  __tsan_unaligned_writex<64>(addr);
}

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_get_current_fiber() {
  return cur_thread();
}

SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_create_fiber(unsigned flags) {
  return FiberCreate(cur_thread(), CALLERPC, flags);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_destroy_fiber(void *fiber) {
  FiberDestroy(cur_thread(), CALLERPC, static_cast<ThreadState *>(fiber));
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_switch_to_fiber(void *fiber, unsigned flags) {
  FiberSwitch(cur_thread(), CALLERPC, static_cast<ThreadState *>(fiber), flags);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_set_fiber_name(void *fiber, const char *name) {
  ThreadSetName(static_cast<ThreadState *>(fiber), name);
}
}  // extern "C"

void __tsan_acquire(void *addr) {
  Acquire(cur_thread(), CALLERPC, (uptr)addr);
}

void __tsan_release(void *addr) {
  Release(cur_thread(), CALLERPC, (uptr)addr);
}
