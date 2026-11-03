//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-device-only -O2 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// Checks which of the two kinds of address space cast reaches the device code.
//
// SYCL 2020 4.7.7.1. leaves the result of the explicit conversion from
// generic_ptr to a narrower address space undefined if the pointer does not
// address that address space, so it must not emit a run-time check.
// sycl::address_space_cast, in contrast, is required to return nullptr in that
// case and therefore does need one.

#include <sycl/sycl.hpp>

namespace as = sycl::access;

using generic_ptr_t =
    sycl::multi_ptr<int, as::address_space::generic_space, as::decorated::yes>;
using global_ptr_t =
    sycl::multi_ptr<int, as::address_space::global_space, as::decorated::yes>;
using local_ptr_t =
    sycl::multi_ptr<int, as::address_space::local_space, as::decorated::yes>;

// The conversion operator is unchecked: no OpGenericCastToPtrExplicit.
//
// CHECK-LABEL: define {{.*}}castViaConversionOpPi
// CHECK-NOT:     generic.cast.to.ptr.explicit
// CHECK:         addrspacecast ptr addrspace(4) {{.*}} to ptr addrspace(1)
// CHECK-NOT:     generic.cast.to.ptr.explicit
// CHECK:       ret
[[clang::sycl_external]] int castViaConversionOp(int *Raw) {
  generic_ptr_t Generic{Raw};
  global_ptr_t Global = static_cast<global_ptr_t>(Generic);
  return *Global;
}

// address_space_cast is checked: it lowers to the SPIR-V instruction and a
// null test.
//
// CHECK-LABEL: define {{.*}}castViaAddressSpaceCastPi
// CHECK:         call {{.*}}@llvm.spv.generic.cast.to.ptr.explicit.p1
// CHECK:       ret
[[clang::sycl_external]] int castViaAddressSpaceCast(int *Raw) {
  auto Global = sycl::address_space_cast<as::address_space::global_space,
                                         as::decorated::yes>(Raw);
  return Global != nullptr ? *Global : 0;
}

// The same for the local address space, which uses a different storage class.
//
// CHECK-LABEL: define {{.*}}castToLocalViaConversionOpPi
// CHECK-NOT:     generic.cast.to.ptr.explicit
// CHECK:         addrspacecast ptr addrspace(4) {{.*}} to ptr addrspace(3)
// CHECK-NOT:     generic.cast.to.ptr.explicit
// CHECK:       ret
[[clang::sycl_external]] int castToLocalViaConversionOp(int *Raw) {
  generic_ptr_t Generic{Raw};
  local_ptr_t Local = static_cast<local_ptr_t>(Generic);
  return *Local;
}

// CHECK-LABEL: define {{.*}}castToLocalViaAddressSpaceCastPi
// CHECK:         call {{.*}}@llvm.spv.generic.cast.to.ptr.explicit.p3
// CHECK:       ret
[[clang::sycl_external]] int castToLocalViaAddressSpaceCast(int *Raw) {
  auto Local = sycl::address_space_cast<as::address_space::local_space,
                                        as::decorated::yes>(Raw);
  return Local != nullptr ? *Local : 0;
}
