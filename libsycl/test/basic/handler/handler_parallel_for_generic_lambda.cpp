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

#include <type_traits>

template <typename KernelName, typename ExpectedType, typename Range>
void testParallelFor(Range R) {
  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<KernelName>(R, [=](auto Item) {
      static_assert(std::is_same<decltype(Item), ExpectedType>::value,
                    "Argument type is unexpected");
    });
  });
}

int main() {
  testParallelFor<class Item1Name, sycl::item<1>>(sycl::range<1>{1});
  testParallelFor<class Item2Name, sycl::item<2>>(sycl::range<2>{1, 1});
  testParallelFor<class Item3Name, sycl::item<3>>(sycl::range<3>{1, 1, 1});
  testParallelFor<class NDItem1Name, sycl::nd_item<1>>(
      sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}});
  testParallelFor<class NDItem2Name, sycl::nd_item<2>>(
      sycl::nd_range<2>{sycl::range<2>{2, 2}, sycl::range<2>{1, 1}});
  testParallelFor<class NDItem3Name, sycl::nd_item<3>>(
      sycl::nd_range<3>{sycl::range<3>{2, 2, 2}, sycl::range<3>{1, 1, 1}});

  sycl::queue Q;
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class GenericInitList1>(sycl::range{1}, [=](auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class GenericInitList2>(sycl::range{1, 1}, [=](auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class GenericInitList3>(sycl::range{1, 1, 1},
                                             [=](auto &) {});
  });

  return 0;
}
