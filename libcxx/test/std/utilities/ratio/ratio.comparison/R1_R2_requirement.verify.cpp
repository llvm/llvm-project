//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <ratio>
//
// [ratio.general]/2
//   Throughout subclause [ratio], the names of template parameters are
//   used to express type requirements. If a template parameter is named
//   R1 or R2, and the template argument is not a specialization of the
//   ratio template, the program is ill-formed.
//
// Since std::ratio_xxx_v uses the same instantiations only one error
// will be generated. These values are tested in a separate test.

#include <ratio>

struct R {
  static const int num = 1;
  static const int den = 1;
};

using r = std::ratio<1, 1>;

namespace equal {
using r_r = std::ratio_equal<r, r>::type;
using R_r = std::ratio_equal<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_equal<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace equal

namespace not_equal {
using r_r = std::ratio_not_equal<r, r>::type;
using R_r = std::ratio_not_equal<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_not_equal<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace not_equal

namespace less {
using r_r = std::ratio_less<r, r>::type;
using R_r = std::ratio_less<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_less<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less

namespace less_equal {
using r_r = std::ratio_less_equal<r, r>::type;
using R_r = std::ratio_less_equal<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_less_equal<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less_equal

namespace greater {
using r_r = std::ratio_greater<r, r>::type;
using R_r = std::ratio_greater<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_greater<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater

namespace greater_equal {
using r_r = std::ratio_greater_equal<r, r>::type;
using R_r =
    std::ratio_greater_equal<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R =
    std::ratio_greater_equal<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater_equal
