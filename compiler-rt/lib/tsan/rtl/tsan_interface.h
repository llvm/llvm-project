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

// User may provide function that would be called right when TSan detects
// an error. The argument 'report' is an opaque pointer that can be used to
// gather additional information using other TSan report API functions.
SANITIZER_INTERFACE_ATTRIBUTE
void __tsan_on_report(void *report);

// If TSan is currently reporting a detected issue on the current thread,
// returns an opaque pointer to the current report. Otherwise returns NULL.
SANITIZER_INTERFACE_ATTRIBUTE
void *__tsan_get_current_report();

// Returns a report's description (issue type), number of duplicate issues
// found, counts of array data (stack traces, memory operations, locations,
// mutexes, threads, unique thread IDs) and a stack trace of a sleep() call (if
// one was involved in the issue).
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_data(void *report, const char **description, int *count,
                           int *stack_count, int *mop_count, int *loc_count,
                           int *mutex_count, int *thread_count,
                           int *unique_tid_count, void **sleep_trace,
                           uptr trace_size);

// Returns information about stack traces included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_stack(void *report, uptr idx, void **trace,
                            uptr trace_size);

// Returns information about memory operations included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_mop(void *report, uptr idx, int *tid, void **addr,
                          int *size, int *write, int *atomic, void **trace,
                          uptr trace_size);

// Returns information about locations included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_loc(void *report, uptr idx, const char **type,
                          void **addr, uptr *start, uptr *size, int *tid,
                          int *fd, int *suppressable, void **trace,
                          uptr trace_size);

// Returns information about mutexes included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_mutex(void *report, uptr idx, uptr *mutex_id, void **addr,
                            int *destroyed, void **trace, uptr trace_size);

// Returns information about threads included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_thread(void *report, uptr idx, int *tid, uptr *pid,
                             int *running, const char **name, int *parent_tid,
                             void **trace, uptr trace_size);

// Returns information about unique thread IDs included in the report.
SANITIZER_INTERFACE_ATTRIBUTE
int __tsan_get_report_unique_tid(void *report, uptr idx, int *tid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TSAN_INTERFACE_H
