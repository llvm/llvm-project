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
// test ratio_multiply

#include <ratio>

struct R {
  static const int num = 1;
  static const int den = 1;
};

using r = std::ratio<1, 1>;

namespace add {
using r_r = std::ratio_add<r, r>::type;
using R_r = std::ratio_add<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_add<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace add

namespace subtract {
using r_r = std::ratio_subtract<r, r>::type;
using R_r = std::ratio_subtract<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_subtract<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace subtract

namespace multiply {
using r_r = std::ratio_multiply<r, r>::type;
using R_r = std::ratio_multiply<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_multiply<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace multiply

namespace divide {
using r_r = std::ratio_divide<r, r>::type;
using R_r = std::ratio_divide<R, r>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using r_R = std::ratio_divide<r, R>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace divide
