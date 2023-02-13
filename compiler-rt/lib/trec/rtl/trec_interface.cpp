//===-- trec_interface.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include "trec_interface.h"

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_ptrauth.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "trec_rtl.h"

#define CALLERPC (StackTrace::GetPreviousInstructionPc((uptr)__builtin_return_address(0)))

using namespace __trec;
using namespace __trec_metadata;

void __trec_init() {
  cur_thread_init();
  Initialize(cur_thread());
}

void __trec_flush_memory() { 
}

#if TREC_HAS_128_BIT
#define GET_128_HIGHHER(x) (uptr)(x >> 64)
#define GET_128_LOWER(x) (uptr)(x)
void __trec_read16(void *addr, bool isPtr, __uint128_t val, void *addr_src_addr,
                   u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, addr_src_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8, isPtr,
             GET_128_LOWER(val), SAI_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8, isPtr,
             GET_128_HIGHHER(val), SAI_addr);
}

void __trec_write16(void *addr, bool isPtr, __uint128_t val,
                    void *addr_src_addr, u16 addr_src_idx, void *val_src_addr,
                    u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, val_src_addr);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8, isPtr,
              GET_128_LOWER(val), SAI_addr, SAI_val);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr + 8, kSizeLog8, isPtr,
              GET_128_HIGHHER(val), SAI_addr, SAI_val);
}
#endif
// __trec_unaligned_read/write calls are emitted by compiler.

void __trec_unaligned_read2(const void *addr, bool isPtr, void *val,
                            void *addr_src_addr, u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, false, false,
                        isPtr, (uptr)val, SAI_addr);
}

void __trec_unaligned_read4(const void *addr, bool isPtr, void *val,
                            void *addr_src_addr, u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, false, false,
                        isPtr, (uptr)val, SAI_addr);
}

void __trec_unaligned_read8(const void *addr, bool isPtr, void *val,
                            void *addr_src_addr, u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, false, false,
                        isPtr, (uptr)val, SAI_addr);
}

#if TREC_HAS_128_BIT
void __trec_unaligned_read16(const void *addr, bool isPtr, __uint128_t val,
                             void *addr_src_addr, u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, false, false,
                        isPtr, GET_128_LOWER(val), SAI_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr + 8, 8, false, false,
                        isPtr, GET_128_HIGHHER(val), SAI_addr);
}
#endif

void __trec_unaligned_write2(void *addr, bool isPtr, void *val,
                             void *addr_src_addr, u16 addr_src_idx,
                             void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 2, true, false,
                        isPtr, (uptr)val, SAI_addr, SAI_val);
}

void __trec_unaligned_write4(void *addr, bool isPtr, void *val,
                             void *addr_src_addr, u16 addr_src_idx,
                             void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 4, true, false,
                        isPtr, (uptr)val, SAI_addr, SAI_val);
}

void __trec_unaligned_write8(void *addr, bool isPtr, void *val,
                             void *addr_src_addr, u16 addr_src_idx,
                             void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, true, false,
                        isPtr, (uptr)val, SAI_addr, SAI_val);
}

#if TREC_HAS_128_BIT
void __trec_unaligned_write16(void *addr, bool isPtr, __uint128_t val,
                              void *addr_src_addr, u16 addr_src_idx,
                              void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, val_src_addr);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr, 8, true, false,
                        isPtr, GET_128_LOWER(val), SAI_addr, SAI_val);
  UnalignedMemoryAccess(cur_thread(), CALLERPC, (uptr)addr + 8, 8, true, false,
                        isPtr, GET_128_HIGHHER(val), SAI_addr, SAI_val);
}
#endif

#undef GET_128_HIGHHER
#undef GET_128_LOWER
// __sanitizer_unaligned_load/store are for user instrumentation.
