// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::div_t;
  using ::ldiv_t;
  using ::lldiv_t;
  using ::size_t;

  // [support.start.term], start and termination
  using ::_Exit;
  using ::abort;
  using ::at_quick_exit;
  using ::atexit;
  using ::exit;
  using ::quick_exit;

  using ::getenv;
  using ::system;

  // [c.malloc], C library memory allocation
  using ::aligned_alloc;
  using ::calloc;
  using ::free;
  using ::malloc;
  using ::realloc;

  using ::atof;
  using ::atoi;
  using ::atol;
  using ::atoll;
  using ::strtod;
  using ::strtof;
  using ::strtol;
  using ::strtold;
  using ::strtoll;
  using ::strtoul;
  using ::strtoull;

  // [c.mb.wcs], multibyte / wide string and character conversion functions
  using ::mblen;
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using ::mbstowcs;
  using ::mbtowc;
  using ::wcstombs;
  using ::wctomb;
#endif
  // [alg.c.library], C standard library algorithms
  using ::bsearch;
  using ::qsort;

  // [c.math.rand], low-quality random number generation
  using ::rand;
  using ::srand;

  // [c.math.abs], absolute values
  using ::abs;

  using ::labs;
  using ::llabs;

  using ::div;
  using ::ldiv;
  using ::lldiv;

} // export
