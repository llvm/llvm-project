//===-- PrintfMatcher.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PrintfMatcher.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdio/printf_core/core_structs.h"

#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {
namespace testing {

using printf_core::FormatFlags;
using printf_core::FormatSection;
using printf_core::LengthModifier;

bool FormatSectionMatcher::match(FormatSection actualValue) {
  actual = actualValue;
  return expected == actual;
}

namespace {

#define IF_FLAG_SHOW_FLAG(flag_name)                                           \
  do {                                                                         \
    if ((form.flags & FormatFlags::flag_name) == FormatFlags::flag_name)       \
      tlog << "\n\t\t" << #flag_name;                                          \
  } while (false)
#define CASE_LM(lm)                                                            \
  case (LengthModifier::lm):                                                   \
    tlog << #lm;                                                               \
    break

static void display(FormatSection form) {
  tlog << "Raw String (len " << form.raw_string.size() << "): \"";
  for (size_t i = 0; i < form.raw_string.size(); ++i) {
    tlog << form.raw_string[i];
  }
  tlog << "\"";
  if (form.has_conv) {
    tlog << "\n\tHas Conv\n\tFlags:";
    IF_FLAG_SHOW_FLAG(LEFT_JUSTIFIED);
    IF_FLAG_SHOW_FLAG(FORCE_SIGN);
    IF_FLAG_SHOW_FLAG(SPACE_PREFIX);
    IF_FLAG_SHOW_FLAG(ALTERNATE_FORM);
    IF_FLAG_SHOW_FLAG(LEADING_ZEROES);
    tlog << "\n";
    tlog << "\tmin width: " << form.min_width << "\n";
    tlog << "\tprecision: " << form.precision << "\n";
    tlog << "\tlength modifier: ";
    switch (form.length_modifier) {
      CASE_LM(none);
      CASE_LM(l);
      CASE_LM(ll);
      CASE_LM(h);
      CASE_LM(hh);
      CASE_LM(j);
      CASE_LM(z);
      CASE_LM(t);
      CASE_LM(L);
    }
    tlog << "\n";
    tlog << "\tconversion name: " << form.conv_name << "\n";
    if (form.conv_name == 'p' || form.conv_name == 'n' || form.conv_name == 's')
      tlog << "\tpointer value: "
           << int_to_hex<uintptr_t>(
                  reinterpret_cast<uintptr_t>(form.conv_val_ptr))
           << "\n";
    else if (form.conv_name != '%')
      tlog << "\tvalue: "
           << int_to_hex<fputil::FPBits<long double>::StorageType>(
                  form.conv_val_raw)
           << "\n";
  }
}
} // anonymous namespace

void FormatSectionMatcher::explainError() {
  tlog << "expected format section: ";
  display(expected);
  tlog << '\n';
  tlog << "actual format section  : ";
  display(actual);
  tlog << '\n';
}

} // namespace testing
} // namespace LIBC_NAMESPACE
