//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++17

// <ratio>
//
// [ratio.general]/2
//   Throughout subclause [ratio], the names of template parameters are
//   used to express type requirements. If a template parameter is named
//   R1 or R2, and the template argument is not a specialization of the
//   ratio template, the program is ill-formed.
//
// Since std::ratio_xxx uses the same instantiations only one error
// will be generated. These types are tested in a separate test.

#include <ratio>

struct R {
  constexpr static int num = 1;
  constexpr static int den = 1;
};

using r = std::ratio<1, 1>;

namespace equal {
constexpr bool r_r_v = std::ratio_equal_v<r, r>;
constexpr bool R_r_v =
    std::ratio_equal_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_equal_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace equal

namespace not_equal {
constexpr bool r_r_v = std::ratio_not_equal_v<r, r>;
constexpr bool R_r_v =
    std::ratio_not_equal_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_not_equal_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace not_equal

namespace less {
constexpr bool r_r_v = std::ratio_less_v<r, r>;
constexpr bool R_r_v =
    std::ratio_less_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_less_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less

namespace less_equal {
constexpr bool r_r_v = std::ratio_less_equal_v<r, r>;
constexpr bool R_r_v =
    std::ratio_less_equal_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_less_equal_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less_equal

namespace greater {
constexpr bool r_r_v = std::ratio_greater_v<r, r>;
constexpr bool R_r_v =
    std::ratio_greater_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_greater_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater

namespace greater_equal {
constexpr bool r_r_v = std::ratio_greater_equal_v<r, r>;
constexpr bool R_r_v =
    std::ratio_greater_equal_v<R, r>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
constexpr bool r_R_v =
    std::ratio_greater_equal_v<r, R>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater_equal
