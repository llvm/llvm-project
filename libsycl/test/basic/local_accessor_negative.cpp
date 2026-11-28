//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl %s -fsyntax-only -DSYCL2020_DISABLE_DEPRECATION_WARNINGS \
// RUN: -Xclang -verify -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

void testDeprecatedAccessorModes(sycl::handler &CGH) {
  // expected-error@*:* {{the accessor specialization with target::local is only available for access_mode::read_write and access_mode::atomic}}
  sycl::accessor<int, 1, sycl::access_mode::read, sycl::target::local> Read(
      1, CGH);

  // expected-error@*:* {{access_mode::atomic is not supported yet for the accessor specialization with target::local}}
  sycl::accessor<int, 1, sycl::access_mode::atomic, sycl::target::local> Atomic(
      1, CGH);

  // expected-error@*:* {{a const qualified DataT is not allowed for the accessor specialization with target::local}}
  sycl::accessor<const int, 1, sycl::access_mode::read_write,
                 sycl::target::local>
      Const(1, CGH);
}

void testDimensionGatedAPI(sycl::handler &CGH) {
  sycl::local_accessor<int, 0> Scalar(CGH);
  // expected-error@+1 {{no matching member function for call to 'get_range'}}
  (void)Scalar.get_range();
  // expected-error@+1 {{no viable overloaded operator[]}}
  (void)Scalar[0];

  sycl::local_accessor<int, 1> Array(1, CGH);
  // expected-error@+1 {{no viable overloaded '='}}
  Array = 1;
  // expected-error@+1 {{no viable conversion}}
  int Value = Array;
  (void)Value;

  // expected-error@+1 {{no matching constructor}}
  sycl::local_accessor<int, 1> NoSize(CGH);
}

void testDeprecatedDimensionGatedAPI(sycl::handler &CGH) {
  sycl::accessor<int, 0, sycl::access_mode::read_write, sycl::target::local>
      Scalar(CGH);
  // expected-error@+1 {{no matching member function for call to 'get_range'}}
  (void)Scalar.get_range();
  // expected-error@+1 {{no viable overloaded operator[]}}
  (void)Scalar[0];

  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>
      Array(1, CGH);
  // expected-error@+1 {{no viable conversion}}
  int Value = Array;
  (void)Value;

  // expected-error@+2 {{no matching constructor}}
  sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>
      NoSize(CGH);
}

void testConstDataT(sycl::handler &CGH) {
  sycl::local_accessor<const int, 0> Scalar(CGH);
  // expected-error@+1 {{no viable overloaded '='}}
  Scalar = 1;

  sycl::local_accessor<const int, 1> ConstArray(1, CGH);
  // expected-error@+1 {{no viable conversion}}
  sycl::local_accessor<int, 1> NonConst = ConstArray;
  (void)NonConst;
}
