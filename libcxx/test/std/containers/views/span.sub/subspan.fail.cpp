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

// template<size_t Offset, size_t Count = dynamic_extent>
//   constexpr span<element_type, see below> subspan() const;
//
// constexpr span<element_type, dynamic_extent> subspan(
//   size_type offset, size_type count = dynamic_extent) const;
//
//  Requires: offset <= size() &&
//            (count == dynamic_extent || count <= size() - offset)

#include <span>

#include "test_macros.h"

constexpr int carr[] = {1, 2, 3, 4};

int main(int, char**) {
  std::span<const int, 4> sp(carr);

  //  Offset too large templatized
  {
    [[maybe_unused]] auto s1 = sp.subspan<5>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Offset out of range in span::subspan()"}}
  }

  //  Count too large templatized
  {
    [[maybe_unused]] auto s1 = sp.subspan<0, 5>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Offset + count out of range in span::subspan()"}}
  }

  //  Offset + Count too large templatized
  {
    [[maybe_unused]] auto s1 = sp.subspan<2, 3>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Offset + count out of range in span::subspan()"}}
  }

  //  Offset + Count overflow templatized
  {
    [[maybe_unused]] auto s1 = sp.subspan<3, std::size_t(-2)>(); // expected-error-re@span:* {{static_assert failed{{( due to requirement '.*')?}} "Offset + count out of range in span::subspan()"}}, expected-error-re@span:* {{array is too large{{(.* elements)}}}}
  }

  return 0;
}
