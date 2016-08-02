//===-- tsan_interface.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// The functions declared in this header will be inserted by the instrumentation
// module.
// This header can be included by the instrumented program or by TSan tests.
//===----------------------------------------------------------------------===//
#ifndef TSAN_INTERFACE_H
#define TSAN_INTERFACE_H

#include <sanitizer_common/sanitizer_internal_defs.h>

// This header should NOT include any other headers.
// All functions in this header are extern "C" and start with __tsan_.

#ifdef __cplusplus
extern "C" {
#endif

// This function should be called at the very beginning of the process,
// before any instrumented code is executed and before any call to malloc.
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_init();

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read1(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read2(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read4(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read8(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read16(void *addr);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write1(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write2(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write4(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write8(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write16(void *addr);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_read2(const void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_read4(const void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_read8(const void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_read16(const void *addr);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_write2(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_write4(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_write8(void *addr);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_unaligned_write16(void *addr);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read1_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read2_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read4_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read8_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_read16_pc(void *addr, void *pc);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write1_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write2_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write4_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write8_pc(void *addr, void *pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_write16_pc(void *addr, void *pc);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_vptr_read(void **vptr_p);
SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_vptr_update(void **vptr_p, void *new_val);

SANITIZER_INTERFACE_ATTRIBUTE void __tsan_func_entry(void *call_pc);
SANITIZER_INTERFACE_ATTRIBUTE void __tsan_func_exit();

SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_read_range(void *addr, unsigned long size);  // NOLINT
SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_write_range(void *addr, unsigned long size);  // NOLINT

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TSAN_INTERFACE_H
