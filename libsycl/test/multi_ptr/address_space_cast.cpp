//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsyntax-only %s -Wno-deprecated-declarations
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only %s \
// RUN:   -Wno-deprecated-declarations

// Checks that a multi_ptr can be cast from a narrower address space to the
// generic one and back, for every combination of address space and decoration,
// and that the conversions between the decorated and the non-decorated
// interface and to multi_ptr<void> are available.

#include <sycl/sycl.hpp>

#include <type_traits>

namespace as = sycl::access;

template <typename DataT, as::address_space Space, as::decorated Decorated>
void checkRoundTrip(DataT *Raw) {
  using multi_ptr_t = sycl::multi_ptr<DataT, Space, Decorated>;
  using generic_ptr_t =
      sycl::multi_ptr<DataT, as::address_space::generic_space, Decorated>;

  multi_ptr_t Narrow = sycl::address_space_cast<Space, Decorated>(Raw);

  // Narrow -> generic, by assignment. Only available when the destination is
  // the generic address space.
  generic_ptr_t Generic;
  Generic = Narrow;

  // Generic -> narrow, by an explicit conversion. SYCL 2020 4.7.7.1. leaves
  // the result undefined if the pointer does not address that address space,
  // so this conversion performs no run-time check.
  multi_ptr_t Back = static_cast<multi_ptr_t>(Generic);

  // The explicit conversion is only available from the generic address space.
  static_assert(std::is_constructible_v<generic_ptr_t, generic_ptr_t>);

  (void)Back;
}

template <typename DataT, as::address_space Space>
void checkDecorations(DataT *Raw) {
  checkRoundTrip<DataT, Space, as::decorated::yes>(Raw);
  checkRoundTrip<DataT, Space, as::decorated::no>(Raw);
}

template <typename DataT, as::address_space Space, as::decorated Decorated>
void checkConversions(DataT *Raw) {
  using multi_ptr_t = sycl::multi_ptr<DataT, Space, Decorated>;
  multi_ptr_t Ptr = sycl::address_space_cast<Space, Decorated>(Raw);

  // Decorated <-> non-decorated.
  sycl::multi_ptr<DataT, Space, as::decorated::yes> Decorated2 = Ptr;
  sycl::multi_ptr<DataT, Space, as::decorated::no> NonDecorated = Ptr;

  // To a multi_ptr of const data.
  sycl::multi_ptr<const DataT, Space, Decorated> Const = Ptr;

  // To multi_ptr<void>, implicitly. Converting back has to be explicit.
  sycl::multi_ptr<void, Space, Decorated> Void = Ptr;
  multi_ptr_t FromVoid = static_cast<multi_ptr_t>(Void);

  // And the const data converts to multi_ptr<const void>.
  sycl::multi_ptr<const void, Space, Decorated> ConstVoid = Const;

  (void)Decorated2;
  (void)NonDecorated;
  (void)FromVoid;
  (void)ConstVoid;
}

[[clang::sycl_external]] void run(int *Raw) {
  checkDecorations<int, as::address_space::global_space>(Raw);
  checkDecorations<int, as::address_space::local_space>(Raw);
  checkDecorations<int, as::address_space::private_space>(Raw);

  checkConversions<int, as::address_space::global_space, as::decorated::yes>(
      Raw);
  checkConversions<int, as::address_space::global_space, as::decorated::no>(
      Raw);
  checkConversions<int, as::address_space::local_space, as::decorated::yes>(
      Raw);

  // make_ptr is the deprecated spelling of address_space_cast.
  auto Made =
      sycl::make_ptr<int, as::address_space::global_space, as::decorated::no>(
          Raw);
  (void)Made;
}
