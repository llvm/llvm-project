//===-- lsan_flags.inc ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LSan runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef LSAN_FLAG
# error "Define LSAN_FLAG prior to including this file!"
#endif

// LSAN_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

LSAN_FLAG(bool, report_objects, false,
          "Print addresses of leaked objects after main leak report.")
LSAN_FLAG(
    int, resolution, 0,
    "Aggregate two objects into one leak if this many stack frames match. If "
    "zero, the entire stack trace must match.")
LSAN_FLAG(int, max_leaks, 0, "The number of leaks reported.")
LSAN_FLAG(bool, detect_double_free, false,
          "Detect calls to free() on an allocation that was already freed. "
          "Validating the pointer is a prerequisite for the check, so while "
          "this is enabled an invalid free is reported as a diagnostic "
          "instead of reaching the allocator and crashing. Stand-alone "
          "LeakSanitizer only: under AddressSanitizer or "
          "HWAddressSanitizer the allocator, and its own double-free "
          "detection, come from that tool instead and this flag does "
          "nothing.")
LSAN_FLAG(int, double_free_max_entries, 65536,
          "Maximum number of freed large allocations remembered for "
          "double-free detection. The oldest entry is evicted once the limit "
          "is reached. Zero means unlimited; values above 4194304 are capped. "
          "Chunks served by the primary allocator are unaffected: they are "
          "tracked in their own metadata.")

// Flags controlling the root set of reachable memory.
LSAN_FLAG(bool, use_globals, true,
          "Root set: include global variables (.data and .bss)")
LSAN_FLAG(bool, use_stacks, true, "Root set: include thread stacks")
LSAN_FLAG(bool, use_registers, true, "Root set: include thread registers")
LSAN_FLAG(bool, use_tls, true,
          "Root set: include TLS and thread-specific storage")
LSAN_FLAG(bool, use_root_regions, true,
          "Root set: include regions added via __lsan_register_root_region().")
LSAN_FLAG(bool, use_ld_allocations, true,
          "Root set: mark as reachable all allocations made from dynamic "
          "linker. This was the old way to handle dynamic TLS, and will "
          "be removed soon. Do not use this flag.")

LSAN_FLAG(bool, use_unaligned, false, "Consider unaligned pointers valid.")
LSAN_FLAG(bool, use_poisoned, false,
          "Consider pointers found in poisoned memory to be valid.")
LSAN_FLAG(bool, use_detached, false,
          "Scan threads even if attaching to them failed.")
LSAN_FLAG(bool, log_pointers, false, "Debug logging")
LSAN_FLAG(bool, log_threads, false, "Debug logging")
LSAN_FLAG(int, tries, 1, "Debug option to repeat leak checking multiple times")
LSAN_FLAG(const char *, suppressions, "", "Suppressions file name.")
LSAN_FLAG(int, thread_suspend_fail, 1,
          "Behaviour if thread suspendion all thread (0 - "
          "abandon leak checking, 1 - continue with leak checking (reported "
          "leaks can be false), 2 - crash (for debugging LSAN)).")