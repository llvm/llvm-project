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

#include <type_traits>

#include <array>
#include <utility>

#if defined(__clang__) && __clang_major__ >= 19
// expected-error@array:* {{could not match _Size against 'type-parameter-0-0'}}
#else
// expected-error@#SA {{template template argument has different template parameters than its corresponding template template parameter}}
#endif
static_assert(!std::__is_specialization_v<std::pair<int, std::size_t>, std::array>); // #SA
