//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__locale>
#include <__locale_dir/support/zos.h> // __locale_guard
#include <memory>
#include <stdlib.h>

_LIBCPP_BEGIN_NAMESPACE_STD

lconv* __libcpp_localeconv_l(locale_t& __l) {
  __locale::__locale_guard __current(__l);

  lconv* lc = localeconv();
  static lconv newlc;
  newlc = *lc;

  enum { max_char_num = 20 };
#define DeepCopy(mbr)                                                                                                  \
  static char buf_##mbr[max_char_num];                                                                                 \
  strncpy(buf_##mbr, lc->mbr, max_char_num);                                                                           \
  newlc.mbr = buf_##mbr;

  DeepCopy(decimal_point);
  DeepCopy(thousands_sep);
  DeepCopy(grouping);
  DeepCopy(int_curr_symbol);
  DeepCopy(currency_symbol);
  DeepCopy(mon_decimal_point);
  DeepCopy(mon_thousands_sep);
  DeepCopy(mon_grouping);
  DeepCopy(positive_sign);
  DeepCopy(negative_sign);
  DeepCopy(__left_parenthesis);
  DeepCopy(__right_parenthesis);

  return &newlc;
}

_LIBCPP_END_NAMESPACE_STD
