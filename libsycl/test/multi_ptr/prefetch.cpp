//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s \
// RUN:   -Wno-deprecated-declarations
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s \
// RUN:   -Wno-deprecated-declarations

// expected-no-diagnostics

// Checks that multi_ptr::prefetch is callable from device code, for both the
// current and the deprecated interface. SYCL 2020 4.7.7.1. makes it an
// implementation-defined optimization that does not affect the functional
// behavior of a kernel, so it is allowed to do nothing.

#include <sycl/sycl.hpp>

namespace as = sycl::access;

[[clang::sycl_external]] void
prefetchDecorated(sycl::multi_ptr<int, as::address_space::global_space,
                                  as::decorated::yes> Ptr) {
  Ptr.prefetch(0);
  Ptr.prefetch(16);
}

[[clang::sycl_external]] void
prefetchNonDecorated(sycl::multi_ptr<int, as::address_space::global_space,
                                     as::decorated::no> Ptr) {
  Ptr.prefetch(0);
}

[[clang::sycl_external]] void
prefetchLegacy(sycl::multi_ptr<int, as::address_space::global_space,
                               as::decorated::legacy> Ptr) {
  Ptr.prefetch(0);
}
