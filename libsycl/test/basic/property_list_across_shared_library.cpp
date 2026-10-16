//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A property is identified by its detail::PropKind, a value both libsycl and
// the code using it read from the same header, so a property list means the
// same thing on both sides of a shared library boundary. This test builds the
// list in a library compiled with hidden visibility, as libsycl itself is
// built, and queries it from the executable.
//
// Identifying a property by an address instead would break this silently:
// has_property() would report false for a property that was passed, and a
// queue constructed with in_order would claim not to be in order. Note that a
// library built with default visibility would not reproduce that, hence
// -fvisibility=hidden below.
//
// RUN: %clangxx -fsycl %fPIC -fvisibility=hidden %shared_lib -DBUILD_LIB %s -o %t.so
// RUN: %clangxx -fsycl %s -o %t.out %t.so
// RUN: %t.out

#include <sycl/sycl.hpp>

// Exported explicitly because the library is built with hidden visibility,
// the way libsycl exports its own API with _LIBSYCL_EXPORT.
#define TEST_EXPORT __attribute__((visibility("default")))

TEST_EXPORT sycl::property_list makeInOrderList();
TEST_EXPORT sycl::property_list makeEmptyList();

#ifdef BUILD_LIB

sycl::property_list makeInOrderList() {
  return sycl::property_list{sycl::property::queue::in_order()};
}

sycl::property_list makeEmptyList() { return sycl::property_list{}; }

#else

#  include <cassert>

int main() {
  const sycl::property_list FromLib = makeInOrderList();

  assert(FromLib.has_property<sycl::property::queue::in_order>() &&
         "in_order set in a shared library must be visible to its user");
  assert(!FromLib.has_property<sycl::property::queue::enable_profiling>() &&
         "enable_profiling was never set, so it must not be reported");

  // Reaching here without throwing is what get_property() promises for a
  // property the list was constructed with.
  FromLib.get_property<sycl::property::queue::in_order>();

  const sycl::property_list Empty = makeEmptyList();
  assert(!Empty.has_property<sycl::property::queue::in_order>());

  // The list is a value: copying it must carry its properties along.
  const sycl::property_list Copy(FromLib);
  assert(Copy.has_property<sycl::property::queue::in_order>());

  return 0;
}

#endif // BUILD_LIB
