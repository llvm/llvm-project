//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// End-to-end coverage of the queue property interface against the real shared
// libsycl. The property list is built here, in user code, while the runtime
// reads it from inside the library, so this also covers a property's identity
// surviving that boundary. The unit tests cannot: they link libsycl's objects
// statically, which hides any disagreement between the two.
//
// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  const sycl::device Dev;

  {
    sycl::queue Q{Dev, sycl::property::queue::in_order()};
    assert(Q.is_in_order() && "queue constructed with in_order must report it");
    assert(Q.has_property<sycl::property::queue::in_order>());
    // Returns a copy of the property; reaching here without throwing is what
    // get_property() promises for a property the queue was constructed with.
    Q.get_property<sycl::property::queue::in_order>();
  }

  {
    sycl::queue Q{Dev};
    assert(!Q.is_in_order());
    assert(!Q.has_property<sycl::property::queue::in_order>());

    bool Thrown = false;
    try {
      Q.get_property<sycl::property::queue::in_order>();
    } catch (const sycl::exception &E) {
      Thrown = E.code() == sycl::make_error_code(sycl::errc::invalid);
    }
    assert(Thrown && "get_property() must throw errc::invalid when absent");
  }

  {
    // A property list is copyable and the copy carries the same properties.
    const sycl::property_list Props{sycl::property::queue::in_order()};
    const sycl::property_list Copy(Props);
    sycl::queue Q{Dev, Copy};
    assert(Q.is_in_order());
  }

  {
    // SYCL 2020 4.6.5.5.: enable_profiling requires aspect::queue_profiling.
    // Queue profiling is not supported yet, so the property is rejected on
    // every device.
    bool Thrown = false;
    try {
      sycl::queue Q{Dev, sycl::property::queue::enable_profiling()};
    } catch (const sycl::exception &E) {
      Thrown =
          E.code() == sycl::make_error_code(sycl::errc::feature_not_supported);
    }
    assert(Thrown && "enable_profiling must throw errc::feature_not_supported");
  }

  return 0;
}
