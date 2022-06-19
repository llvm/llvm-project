//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// template <class T, class Alloc>
//   void swap(vector<T,Alloc>& x, vector<T,Alloc>& y);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <vector>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::vector<operator_hijacker> vo;
  std::vector<operator_hijacker> v;
  v.swap(vo);
}
