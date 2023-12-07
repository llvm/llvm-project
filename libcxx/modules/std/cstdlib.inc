// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
  using std::div_t;
  using std::ldiv_t;
  using std::lldiv_t;
  using std::size_t;

  // [support.start.term], start and termination
  using std::_Exit;
  using std::abort;
  using std::at_quick_exit;
  using std::atexit;
  using std::exit;
  using std::quick_exit;

  using std::getenv;
  using std::system;

  // [c.malloc], C library memory allocation
  using std::aligned_alloc;
  using std::calloc;
  using std::free;
  using std::malloc;
  using std::realloc;

  using std::atof;
  using std::atoi;
  using std::atol;
  using std::atoll;
  using std::strtod;
  using std::strtof;
  using std::strtol;
  using std::strtold;
  using std::strtoll;
  using std::strtoul;
  using std::strtoull;

  // [c.mb.wcs], multibyte / wide string and character conversion functions
  using std::mblen;
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::mbstowcs;
  using std::mbtowc;
  using std::wcstombs;
  using std::wctomb;
#endif
  // [alg.c.library], C standard library algorithms
  using std::bsearch;
  using std::qsort;

  // [c.math.rand], low-quality random number generation
  using std::rand;
  using std::srand;

  // [c.math.abs], absolute values
  using std::abs;

  using std::labs;
  using std::llabs;

  using std::div;
  using std::ldiv;
  using std::lldiv;
} // namespace std
