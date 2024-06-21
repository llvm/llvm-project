//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

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
  constexpr static int num = 1;
  constexpr static int den = 1;
};

using valid = std::ratio<1, 1>;

void test() {
  // equal
  (void)std::ratio_equal_v<valid, valid>;
  (void)std::ratio_equal_v<invalid, valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
  (void)std::ratio_equal_v<valid, invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}

  // not_equal
  (void)std::ratio_not_equal_v<valid, valid>;
  (void)std::ratio_not_equal_v<invalid,
                               valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
  (void)std::ratio_not_equal_v<valid,
                               invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}

  // less
  (void)std::ratio_less_v<valid, valid>;
  (void)std::ratio_less_v<invalid, valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
  (void)std::ratio_less_v<valid, invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}

  // less_equal
  (void)std::ratio_less_equal_v<valid, valid>;
  (void)std::ratio_less_equal_v<invalid,
                                valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
  (void)std::ratio_less_equal_v<valid,
                                invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}

  // greater
  (void)std::ratio_greater_v<valid, valid>;
  (void)std::ratio_greater_v<invalid, valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}
  (void)std::ratio_greater_v<valid, invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}

  // greater_equal
  (void)std::ratio_greater_equal_v<valid, valid>;

  (void)std::ratio_greater_equal_v<invalid,
                                   valid>; // expected-error@*:* {{R1 to be a specialisation of the ratio template}}

  (void)std::ratio_greater_equal_v<valid,
                                   invalid>; // expected-error@*:* {{R2 to be a specialisation of the ratio template}}
}
