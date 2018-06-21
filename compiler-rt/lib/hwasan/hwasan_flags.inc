//===-- hwasan_flags.inc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hwasan runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef HWASAN_FLAG
# error "Define HWASAN_FLAG prior to including this file!"
#endif

// HWASAN_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

HWASAN_FLAG(bool, tag_in_malloc, true, "")
HWASAN_FLAG(bool, tag_in_free, true, "")
HWASAN_FLAG(bool, retag_in_realloc, true, "")
HWASAN_FLAG(bool, print_stats, false, "")
HWASAN_FLAG(bool, halt_on_error, true, "")
HWASAN_FLAG(bool, atexit, false, "")

// Test only flag to disable malloc/realloc/free memory tagging on startup.
// Tagging can be reenabled with __hwasan_enable_allocator_tagging().
HWASAN_FLAG(bool, disable_allocator_tagging, false, "")
