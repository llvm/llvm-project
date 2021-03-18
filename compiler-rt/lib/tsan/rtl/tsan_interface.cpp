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
#include "tsan_interface_ann.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_ptrauth.h"

#define CALLERPC ((uptr)__builtin_return_address(0))

using namespace __tsan;

void __tsan_init() {
  cur_thread_init();
  Initialize(cur_thread());
}

void __tsan_flush_memory() {
  FlushShadowMemory();
}

void __tsan_read16(void *addr) {
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8);
}

void __tsan_write16(void *addr) {
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8);
}

void __tsan_read16_pc(void *addr, void *pc) {
  MemoryRead(cur_thread(), STRIP_PC(pc), (uptr)addr, kSizeLog8);
  MemoryRead(cur_thread(), STRIP_PC(pc), (uptr)addr + 8, kSizeLog8);
}

void __tsan_write16_pc(void *addr, void *pc) {
  MemoryWrite(cur_thread(), STRIP_PC(pc), (uptr)addr, kSizeLog8);
  MemoryWrite(cur_thread(), STRIP_PC(pc), (uptr)addr + 8, kSizeLog8);
}

// __tsan_unaligned_read/write calls are emitted by compiler.

void __tsan_unaligned_read2(const void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, false, false);
}

void __tsan_unaligned_read4(const void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, false, false);
}

void __tsan_unaligned_read8(const void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, false, false);
}

void __tsan_unaligned_read16(const void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 16, false, false);
}

void __tsan_unaligned_write2(void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, true, false);
}

void __tsan_unaligned_write4(void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, true, false);
}

void __tsan_unaligned_write8(void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, true, false);
}

void __tsan_unaligned_write16(void *addr) {
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 16, true, false);
}

// __sanitizer_unaligned_load/store are for user instrumentation.

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
u16 __sanitizer_unaligned_load16(const uu16 *addr) {
  __tsan_unaligned_read2(addr);
  return *addr;
}

SANITIZER_INTERFACE_ATTRIBUTE
u32 __sanitizer_unaligned_load32(const uu32 *addr) {
  __tsan_unaligned_read4(addr);
  return *addr;
}

SANITIZER_INTERFACE_ATTRIBUTE
u64 __sanitizer_unaligned_load64(const uu64 *addr) {
  __tsan_unaligned_read8(addr);
  return *addr;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store16(uu16 *addr, u16 v) {
  __tsan_unaligned_write2(addr);
  *addr = v;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store32(uu32 *addr, u32 v) {
  __tsan_unaligned_write4(addr);
  *addr = v;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_unaligned_store64(uu64 *addr, u64 v) {
  __tsan_unaligned_write8(addr);
  *addr = v;
}

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
