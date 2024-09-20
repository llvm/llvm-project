//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template <class _Tp, template <class...> class _Template>
// inline constexpr bool __is_specialization_v = true if and only if _Tp is a specialization of _Template
//
// Tests the ill-formed instantiations.

#include <__type_traits/is_specialization.h>
#include <array>
#include <utility>

// expected-error-re@*:* {{{{could not match _Size against 'type-parameter-0-0'|different template parameters}}}}
static_assert(!std::__is_specialization_v<std::pair<int, std::size_t>, std::array>);
