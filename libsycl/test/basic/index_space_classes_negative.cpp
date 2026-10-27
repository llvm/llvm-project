//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -fsyntax-only -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

void testInvalidDimensionGatedAPI() {
  sycl::id<2> Id2(1, 2);
  std::size_t ScalarFromId2 = Id2; // expected-error {{no viable conversion}}
  (void)ScalarFromId2;

  sycl::id<2> BadId2(7);          // expected-error {{no matching constructor}}
  sycl::range<3> BadRange3(1, 2); // expected-error {{no matching constructor}}

  (void)BadId2;
  (void)BadRange3;
}
