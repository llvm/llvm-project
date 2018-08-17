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

// If false, use simple increment of a thread local counter to generate new
// tags.
HWASAN_FLAG(bool, random_tags, true, "")

HWASAN_FLAG(
    int, max_malloc_fill_size, 0x1000,  // By default, fill only the first 4K.
    "HWASan allocator flag. max_malloc_fill_size is the maximal amount of "
    "bytes that will be filled with malloc_fill_byte on malloc.")
HWASAN_FLAG(
    int, max_free_fill_size, 0,
    "HWASan allocator flag. max_free_fill_size is the maximal amount of "
    "bytes that will be filled with free_fill_byte during free.")
HWASAN_FLAG(int, malloc_fill_byte, 0xbe,
          "Value used to fill the newly allocated memory.")
HWASAN_FLAG(int, free_fill_byte, 0x55,
          "Value used to fill deallocated memory.")
