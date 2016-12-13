//===-- sanitizer/coverage_interface.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Public interface for sanitizer coverage.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_COVERAG_INTERFACE_H
#define SANITIZER_COVERAG_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

  // Initialize coverage.
  void __sanitizer_cov_init();
  // Record and dump coverage info.
  void __sanitizer_cov_dump();

  //  Dump collected coverage info. Sorts pcs by module into individual
  //  .sancov files.
  void __sanitizer_dump_coverage(const uintptr_t *pcs, uintptr_t len);

  // Open <name>.sancov.packed in the coverage directory and return the file
  // descriptor. Returns -1 on failure, or if coverage dumping is disabled.
  // This is intended for use by sandboxing code.
  intptr_t __sanitizer_maybe_open_cov_file(const char *name);
  // Get the number of unique covered blocks (or edges).
  // This can be useful for coverage-directed in-process fuzzers.
  uintptr_t __sanitizer_get_total_unique_coverage();
  // Get the number of unique indirect caller-callee pairs.
  uintptr_t __sanitizer_get_total_unique_caller_callee_pairs();

  // Reset the basic-block (edge) coverage to the initial state.
  // Useful for in-process fuzzing to start collecting coverage from scratch.
  // Experimental, will likely not work for multi-threaded process.
  void __sanitizer_reset_coverage();
  // Set *data to the array of covered PCs and return the size of that array.
  // Some of the entries in *data will be zero.
  uintptr_t __sanitizer_get_coverage_guards(uintptr_t **data);

  // The coverage instrumentation may optionally provide imprecise counters.
  // Rather than exposing the counter values to the user we instead map
  // the counters to a bitset.
  // Every counter is associated with 8 bits in the bitset.
  // We define 8 value ranges: 1, 2, 3, 4-7, 8-15, 16-31, 32-127, 128+
  // The i-th bit is set to 1 if the counter value is in the i-th range.
  // This counter-based coverage implementation is *not* thread-safe.

  // Returns the number of registered coverage counters.
  uintptr_t __sanitizer_get_number_of_counters();
  // Updates the counter 'bitset', clears the counters and returns the number of
  // new bits in 'bitset'.
  // If 'bitset' is nullptr, only clears the counters.
  // Otherwise 'bitset' should be at least
  // __sanitizer_get_number_of_counters bytes long and 8-aligned.
  uintptr_t
  __sanitizer_update_counter_bitset_and_clear_counters(uint8_t *bitset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_COVERAG_INTERFACE_H
