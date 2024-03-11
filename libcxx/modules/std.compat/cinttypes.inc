// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::imaxdiv_t;

  using ::imaxabs;
  using ::imaxdiv;
  using ::strtoimax;
  using ::strtoumax;
  using ::wcstoimax;
  using ::wcstoumax;

  // abs is conditionally here, but always present in cmath.cppm. To avoid
  // conflicing declarations omit the using here.

  // div is conditionally here, but always present in cstdlib.cppm. To avoid
  // conflicing declarations omit the using here.
} // export
