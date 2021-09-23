//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>
//
// template <class T, class Alloc, class... Args>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   Args&&... args) noexcept;
// template <class T, class Alloc, class Tuple1, class Tuple2>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc, piecewise_construct_t,
//                                                   Tuple1&& x, Tuple2&& y)
//                                                   noexcept;
// template <class T, class Alloc>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   U&& u, V&& v) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   const pair<U, V>& pr) noexcept;
// template <class T, class Alloc, class U, class V>
//   constexpr auto uses_allocator_construction_args(const Alloc& alloc,
//                                                   pair<U, V>&& pr) noexcept;

#include <memory>

struct UsesNotConstructible {
    using allocator_type = std::allocator<int>;
    UsesNotConstructible() = default;
};

struct NotUses {};

int main(int, char**) {
    std::allocator<int> alloc;
    {
        // expected-error-re@__memory/uses_allocator_utils.h:* 1 {{static_assert failed{{.*}} "T uses allocator but is not constructible using one of the convensions"}}
        auto args = std::uses_allocator_construction_args<UsesNotConstructible>(alloc);
    }
    {
        // expected-error-re@__memory/uses_allocator_utils.h:* 1 {{static_assert failed{{.*}} "T not uses allocator and not constructible from Args"}}
        auto args = std::uses_allocator_construction_args<NotUses>(alloc, 1, 2, 3);
    }
}