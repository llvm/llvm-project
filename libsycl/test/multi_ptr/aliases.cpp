//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s \
// RUN:   -Wno-deprecated-declarations \
// RUN:   -Xclang -verify-ignore-unexpected=note,warning
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s \
// RUN:   -Wno-deprecated-declarations \
// RUN:   -Xclang -verify-ignore-unexpected=note,warning

// expected-no-diagnostics

// Checks that the explicit pointer aliases of SYCL 2020 4.7.7.2. name the
// expected specialization of sycl::multi_ptr.

#include <sycl/sycl.hpp>

#include <type_traits>

namespace as = sycl::access;

// The aliases that take a decoration default to access::decorated::legacy.
template <typename DataT, as::decorated Decorated>
void checkAddressSpaceAliases() {
  static_assert(
      std::is_same_v<
          sycl::global_ptr<DataT, Decorated>,
          sycl::multi_ptr<DataT, as::address_space::global_space, Decorated>>);
  static_assert(
      std::is_same_v<
          sycl::local_ptr<DataT, Decorated>,
          sycl::multi_ptr<DataT, as::address_space::local_space, Decorated>>);
  static_assert(
      std::is_same_v<
          sycl::private_ptr<DataT, Decorated>,
          sycl::multi_ptr<DataT, as::address_space::private_space, Decorated>>);
  static_assert(
      std::is_same_v<
          sycl::generic_ptr<DataT, Decorated>,
          sycl::multi_ptr<DataT, as::address_space::generic_space, Decorated>>);
}

template <typename DataT> void checkAliases() {
  checkAddressSpaceAliases<DataT, as::decorated::yes>();
  checkAddressSpaceAliases<DataT, as::decorated::no>();
  checkAddressSpaceAliases<DataT, as::decorated::legacy>();

  // Omitting the decoration selects the deprecated interface.
  static_assert(std::is_same_v<sycl::global_ptr<DataT>,
                               sycl::global_ptr<DataT, as::decorated::legacy>>);
  static_assert(std::is_same_v<sycl::local_ptr<DataT>,
                               sycl::local_ptr<DataT, as::decorated::legacy>>);
  static_assert(
      std::is_same_v<sycl::private_ptr<DataT>,
                     sycl::private_ptr<DataT, as::decorated::legacy>>);
  static_assert(
      std::is_same_v<sycl::generic_ptr<DataT>,
                     sycl::generic_ptr<DataT, as::decorated::legacy>>);

  // constant_ptr has no decoration parameter and is always legacy.
  static_assert(
      std::is_same_v<sycl::constant_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::constant_space,
                                     as::decorated::legacy>>);

  // raw_* exposes a non-decorated pointer.
  static_assert(
      std::is_same_v<sycl::raw_global_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::global_space,
                                     as::decorated::no>>);
  static_assert(
      std::is_same_v<sycl::raw_local_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::local_space,
                                     as::decorated::no>>);
  static_assert(
      std::is_same_v<sycl::raw_private_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::private_space,
                                     as::decorated::no>>);
  static_assert(
      std::is_same_v<sycl::raw_generic_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::generic_space,
                                     as::decorated::no>>);

  // decorated_* exposes a decorated pointer.
  static_assert(
      std::is_same_v<sycl::decorated_global_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::global_space,
                                     as::decorated::yes>>);
  static_assert(
      std::is_same_v<sycl::decorated_local_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::local_space,
                                     as::decorated::yes>>);
  static_assert(
      std::is_same_v<sycl::decorated_private_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::private_space,
                                     as::decorated::yes>>);
  static_assert(
      std::is_same_v<sycl::decorated_generic_ptr<DataT>,
                     sycl::multi_ptr<DataT, as::address_space::generic_space,
                                     as::decorated::yes>>);

  // The raw_* and decorated_* aliases of the same address space differ only in
  // the decoration, so they are distinct types.
  static_assert(!std::is_same_v<sycl::raw_global_ptr<DataT>,
                                sycl::decorated_global_ptr<DataT>>);
}

template void checkAliases<int>();
template void checkAliases<const int>();
template void checkAliases<void>();
template void checkAliases<const void>();

// The aliases are usable as types, not just comparable.
[[clang::sycl_external]] void use(int *Raw) {
  sycl::raw_global_ptr<int> Global =
      sycl::address_space_cast<as::address_space::global_space,
                               as::decorated::no>(Raw);
  sycl::decorated_global_ptr<int> Decorated = Global;
  sycl::global_ptr<int> Legacy{Raw};
  sycl::raw_generic_ptr<int> Generic{Raw};

  (void)Global;
  (void)Decorated;
  (void)Legacy;
  (void)Generic;
}
