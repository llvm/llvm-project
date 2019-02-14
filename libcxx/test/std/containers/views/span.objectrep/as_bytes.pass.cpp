// -*- C++ -*-
//===------------------------------ span ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <span>

// template <class ElementType, ptrdiff_t Extent>
//     span<const byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : static_cast<ptrdiff_t>(sizeof(ElementType)) * Extent>
//     as_bytes(span<ElementType, Extent> s) noexcept;


#include <span>
#include <cassert>
#include <string>

#include "test_macros.h"

template<typename Span>
void testRuntimeSpan(Span sp)
{
    ASSERT_NOEXCEPT(std::as_bytes(sp));

    auto spBytes = std::as_bytes(sp);
    using SB = decltype(spBytes);
    ASSERT_SAME_TYPE(const std::byte, typename SB::element_type);

    if (sp.extent == std::dynamic_extent)
        assert(spBytes.extent == std::dynamic_extent);
    else
        assert(spBytes.extent == static_cast<std::ptrdiff_t>(sizeof(typename Span::element_type)) * sp.extent);

    assert((void *) spBytes.data() == (void *) sp.data());
    assert(spBytes.size() == sp.size_bytes());
}

struct A{};
int iArr2[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};

int main(int, char**)
{
    testRuntimeSpan(std::span<int>        ());
    testRuntimeSpan(std::span<long>       ());
    testRuntimeSpan(std::span<double>     ());
    testRuntimeSpan(std::span<A>          ());
    testRuntimeSpan(std::span<std::string>());

    testRuntimeSpan(std::span<int, 0>        ());
    testRuntimeSpan(std::span<long, 0>       ());
    testRuntimeSpan(std::span<double, 0>     ());
    testRuntimeSpan(std::span<A, 0>          ());
    testRuntimeSpan(std::span<std::string, 0>());

    testRuntimeSpan(std::span<int>(iArr2, 1));
    testRuntimeSpan(std::span<int>(iArr2, 2));
    testRuntimeSpan(std::span<int>(iArr2, 3));
    testRuntimeSpan(std::span<int>(iArr2, 4));
    testRuntimeSpan(std::span<int>(iArr2, 5));

    testRuntimeSpan(std::span<int, 1>(iArr2 + 5, 1));
    testRuntimeSpan(std::span<int, 2>(iArr2 + 4, 2));
    testRuntimeSpan(std::span<int, 3>(iArr2 + 3, 3));
    testRuntimeSpan(std::span<int, 4>(iArr2 + 2, 4));
    testRuntimeSpan(std::span<int, 5>(iArr2 + 1, 5));

    std::string s;
    testRuntimeSpan(std::span<std::string>(&s, (std::ptrdiff_t) 0));
    testRuntimeSpan(std::span<std::string>(&s, 1));

  return 0;
}
