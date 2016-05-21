//===-- esan_flags.inc ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Esan runtime flags.
//
//===----------------------------------------------------------------------===//

#ifndef ESAN_FLAG
# error "Define ESAN_FLAG prior to including this file!"
#endif

// ESAN_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

// Cross-tool options:
ESAN_FLAG(int, cache_line_size, 64,
          "The number of bytes in a cache line.  For the working-set tool, this "
          "cannot be changed without also changing the compiler "
          "instrumentation.")
