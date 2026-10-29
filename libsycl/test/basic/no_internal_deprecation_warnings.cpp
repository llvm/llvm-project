//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Headers found through -I are not system headers, so diagnostics raised
// inside them are reported. Using only non-deprecated APIs must stay
// warning-free in that mode, while deprecated APIs must still warn at the
// point of use.

// RUN: %clangxx -fsycl %sycl_nonsystem_include %s -fsyntax-only \
// RUN: -Wno-error=deprecated-declarations -Xclang -verify \
// RUN: -Xclang -verify-ignore-unexpected=note

#include <sycl/sycl.hpp>

template <int Dims> void useNdItem(sycl::nd_item<Dims> It) {
  sycl::nd_range<Dims> NdRange = It.get_nd_range();
  (void)NdRange;
}

void testNoInternalDeprecationWarnings(sycl::queue &Q) {
  Q.parallel_for<class NdItem1>(sycl::nd_range<1>{{4}, {2}},
                                [](sycl::nd_item<1> It) { useNdItem(It); });
  Q.parallel_for<class NdItem2>(sycl::nd_range<2>{{4, 4}, {2, 2}},
                                [](sycl::nd_item<2> It) { useNdItem(It); });
  Q.parallel_for<class NdItem3>(sycl::nd_range<3>{{4, 4, 4}, {2, 2, 2}},
                                [](sycl::nd_item<3> It) { useNdItem(It); });
}

void testDeprecatedAPIsStillWarn(sycl::queue &Q) {
  sycl::nd_range<1> NdRange({4}, {2}, {1}); // expected-warning {{deprecated}}
  (void)NdRange.get_offset();               // expected-warning {{deprecated}}

  Q.parallel_for<class NdItemOffset>(
      sycl::nd_range<1>{{4}, {2}}, [](sycl::nd_item<1> It) {
        (void)It.get_offset(); // expected-warning {{deprecated}}
      });
  Q.parallel_for<class ItemOffset>(sycl::range<1>{4}, [](sycl::item<1> It) {
    (void)It.get_offset(); // expected-warning {{deprecated}}
  });
}
