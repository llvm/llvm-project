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
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SANITIZER_COVERAG_INTERFACE_H
