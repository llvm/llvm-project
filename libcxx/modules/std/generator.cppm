// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<generator>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <generator>
#endif

export module std:generator;
export namespace std {
#if 0
  using std::generator;
#endif
} // namespace std
