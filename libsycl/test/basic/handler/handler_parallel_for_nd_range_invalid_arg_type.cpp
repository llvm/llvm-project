//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=error,note %s

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class HandlerNDRangeInvalidArgType>(
        sycl::nd_range<1>{sycl::range<1>{4}, sycl::range<1>{2}},
        [=](sycl::item<1>) {}); // expected-error@* {{must be sycl::nd_item or
                                // be convertible from sycl::nd_item}}
  });

  return 0;
}
