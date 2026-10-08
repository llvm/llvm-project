//===-- PrintfMatcher.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PrintfMatcher.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/__support/printf_core/core_structs.h"
#include "src/__support/printf_core/printf_config.h"

#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

using printf_core::BasicFormatSection;
using printf_core::FormatFlags;
using printf_core::LengthModifier;

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
#define CASE_LM_BIT_WIDTH(lm, bw)                                              \
  case (LengthModifier::lm):                                                   \
    tlog << #lm << "\n\tbit width: :" << bw;                                   \
    break

template <typename CharT> void display_impl(BasicFormatSection<CharT> form) {
  tlog << "Raw String (len " << form.raw_string.size() << "): \"";
  cpp::string raw_string_utf8;
  if constexpr (cpp::is_same_v<CharT, char>) {
    raw_string_utf8 = form.raw_string;
  } else {
    raw_string_utf8 = try_convert_to_utf8(form.raw_string);
  }
  for (size_t i = 0; i < raw_string_utf8.size(); ++i) {
    tlog << raw_string_utf8[i];
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
#if defined(LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128)
      CASE_LM(Q);
#endif // LIBC_INTERNAL_PRINTF_CONVERT_FLOAT128
#ifndef LIBC_COPT_PRINTF_DISABLE_BITINT
      CASE_LM_BIT_WIDTH(w, form.bit_width);
      CASE_LM_BIT_WIDTH(wf, form.bit_width);
#endif // LIBC_COPT_PRINTF_DISABLE_BITINT
    }

    cpp::string conv_name_utf8;
    if constexpr (cpp::is_same_v<CharT, char>) {
      conv_name_utf8 = cpp::string(&form.conv_name, 1);
    } else {
      conv_name_utf8 =
          try_convert_to_utf8(cpp::wstring_view(&form.conv_name, 1));
    }
    tlog << "\n";
    tlog << "\tconversion name: " << conv_name_utf8 << "\n";
    if (conv_name_utf8 == "p" || conv_name_utf8 == "n" || conv_name_utf8 == "s")
      tlog << "\tpointer value: "
           << int_to_hex<uintptr_t>(
                  reinterpret_cast<uintptr_t>(form.conv_val_ptr))
           << "\n";
    else if (conv_name_utf8 != "%")
      tlog << "\tvalue: " << int_to_hex(form.conv_val_raw) << "\n";
  }
}

} // anonymous namespace

void display(const BasicFormatSection<char> &format_section) {
  display_impl(format_section);
}

void display(const BasicFormatSection<wchar_t> &format_section) {
  display_impl(format_section);
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
