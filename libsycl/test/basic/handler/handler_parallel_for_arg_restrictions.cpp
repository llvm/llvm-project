//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

template <int Dims> struct ConvertibleFromNDItem {
  ConvertibleFromNDItem(sycl::nd_item<Dims>) {}
};

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1}, [=](sycl::item<1>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1, 1}, [=](sycl::item<2>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1, 1, 1}, [=](sycl::item<3>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1}, [=](sycl::id<1>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{4}, sycl::range{2}},
                     [=](sycl::nd_item<1>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{2, 4}, sycl::range{1, 2}},
                     [=](sycl::nd_item<2>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{2, 2, 4}, sycl::range{1, 1, 2}},
                     [=](sycl::nd_item<3>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{4}, sycl::range{2}},
                     [=](ConvertibleFromNDItem<1>) {});
  });

  return 0;
}
