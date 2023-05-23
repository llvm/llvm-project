// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <scoped_allocator>

export module std:scoped_allocator;
export namespace std {
  // class template scoped_allocator_adaptor
  using std::scoped_allocator_adaptor;

  // [scoped.adaptor.operators], scoped allocator operators
  using std::operator==;
#if 1 // P1614
  using std::operator!=;
#endif

} // namespace std
