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
// Since all std::ratio_xxx_v variables use the same instantiation, only one
// error will be generated. These values are tested in a separate test.

#include <ratio>

struct invalid {
  static const int num = 1;
  static const int den = 1;
};

using valid = std::ratio<1, 1>;

namespace equal {
using valid_valid = std::ratio_equal<valid, valid>::type;
using invalid_valid =
    std::ratio_equal<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_equal<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace equal

namespace not_equal {
using valid_valid = std::ratio_not_equal<valid, valid>::type;
using invalid_valid =
    std::ratio_not_equal<invalid,
                         valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_not_equal<valid,
                         invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace not_equal

namespace less {
using valid_valid = std::ratio_less<valid, valid>::type;
using invalid_valid =
    std::ratio_less<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_less<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less

namespace less_equal {
using valid_valid = std::ratio_less_equal<valid, valid>::type;
using invalid_valid =
    std::ratio_less_equal<invalid,
                          valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_less_equal<valid,
                          invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace less_equal

namespace greater {
using valid_valid = std::ratio_greater<valid, valid>::type;
using invalid_valid =
    std::ratio_greater<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_greater<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater

namespace greater_equal {
using valid_valid = std::ratio_greater_equal<valid, valid>::type;
using invalid_valid =
    std::ratio_greater_equal<invalid,
                             valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_greater_equal<valid,
                             invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace greater_equal
