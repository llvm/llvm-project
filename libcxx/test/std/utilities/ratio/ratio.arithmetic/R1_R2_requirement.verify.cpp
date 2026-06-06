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

#include <ratio>

struct invalid {
  static const int num = 1;
  static const int den = 1;
};

using valid = std::ratio<1, 1>;

namespace add {
using valid_valid = std::ratio_add<valid, valid>::type;
using invalid_valid =
    std::ratio_add<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_add<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace add

namespace subtract {
using valid_valid = std::ratio_subtract<valid, valid>::type;
using invalid_valid =
    std::ratio_subtract<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_subtract<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace subtract

namespace multiply {
using valid_valid = std::ratio_multiply<valid, valid>::type;
using invalid_valid =
    std::ratio_multiply<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_multiply<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace multiply

namespace divide {
using valid_valid = std::ratio_divide<valid, valid>::type;
using invalid_valid =
    std::ratio_divide<invalid, valid>::type; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
using valid_invalid =
    std::ratio_divide<valid, invalid>::type; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
} // namespace divide
