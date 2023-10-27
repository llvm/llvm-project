//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template<class OtherElementType, class OtherExtents,
//         class OtherLayoutPolicy, class OtherAccessor>
//  constexpr explicit(see below)
//    mdspan(const mdspan<OtherElementType, OtherExtents,
//                        OtherLayoutPolicy, OtherAccessor>& other);
//
// Constraints:
//   - is_constructible_v<mapping_type, const OtherLayoutPolicy::template mapping<OtherExtents>&> is true, and
//   - is_constructible_v<accessor_type, const OtherAccessor&> is true.
// Mandates:
//   - is_constructible_v<data_handle_type, const OtherAccessor::data_handle_type&> is
//   - is_constructible_v<extents_type, OtherExtents> is true.
//
// Preconditions:
//   - For each rank index r of extents_type, static_extent(r) == dynamic_extent || static_extent(r) == other.extent(r) is true.
//   - [0, map_.required_span_size()) is an accessible range of ptr_ and acc_ for values of ptr_, map_, and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with other.ptr_,
//   - direct-non-list-initializes map_ with other.map_, and
//   - direct-non-list-initializes acc_ with other.acc_.
//
// Remarks: The expression inside explicit is equivalent to:
//   !is_convertible_v<const OtherLayoutPolicy::template mapping<OtherExtents>&, mapping_type>
//   || !is_convertible_v<const OtherAccessor&, accessor_type>

#include <mdspan>
#include "CustomTestAccessors.h"
#include "CustomTestLayouts.h"

void cant_construct_data_handle_type() {
  int data;
  std::mdspan<int, std::extents<int>, std::layout_right, convertible_accessor_but_not_handle<int>> m_nc(&data);
  // expected-error-re@*:* {{{{.*}}no matching constructor for initialization of {{.*}} (aka 'not_const_convertible_handle<const int>')}}
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: incompatible data_handle_type for mdspan construction}}
  [[maybe_unused]] std::
      mdspan<const int, std::extents<int>, std::layout_right, convertible_accessor_but_not_handle<const int>>
          m_c(m_nc);
}

void mapping_constructible_despite_extents_compatibility() {
  int data;
  std::mdspan<int, std::extents<int>, always_convertible_layout> m(&data);
  // expected-error-re@*:* {{static assertion failed {{.*}}mdspan: incompatible extents for mdspan construction}}
  [[maybe_unused]] std::mdspan<int, std::extents<int, 5>, always_convertible_layout> m2(m);
}
