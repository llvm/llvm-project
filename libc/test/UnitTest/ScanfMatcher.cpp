//===-- ScanfMatcher.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScanfMatcher.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/stdio/scanf_core/core_structs.h"

#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace testing {

using scanf_core::FormatFlags;
using scanf_core::FormatSection;
using scanf_core::LengthModifier;

namespace {

#define IF_FLAG_SHOW_FLAG(flag_name)                                           \
  do {                                                                         \
    if ((form.flags & FormatFlags::flag_name) == FormatFlags::flag_name)       \
      listener << "\n\t\t" << #flag_name;                                      \
  } while (false)
#define CASE_LM(lm)                                                            \
  case (LengthModifier::lm):                                                   \
    listener << #lm;                                                           \
    break

void display(MatchResultListener &listener, const FormatSection &form) {
  listener << "Raw String (len " << form.raw_string.size() << "): \"";
  for (size_t i = 0; i < form.raw_string.size(); ++i) {
    listener << form.raw_string[i];
  }
  listener << "\"";
  if (form.has_conv) {
    listener << "\n\tHas Conv\n\tFlags:";
    IF_FLAG_SHOW_FLAG(NO_WRITE);
    IF_FLAG_SHOW_FLAG(ALLOCATE);
    listener << "\n";
    listener << "\tmax width: " << form.max_width << "\n";
    listener << "\tlength modifier: ";
    switch (form.length_modifier) {
      CASE_LM(NONE);
      CASE_LM(l);
      CASE_LM(ll);
      CASE_LM(h);
      CASE_LM(hh);
      CASE_LM(j);
      CASE_LM(z);
      CASE_LM(t);
      CASE_LM(L);
    }
    listener << "\n";
    // If the pointer is used (NO_WRITE is not set and the conversion isn't %).
    if (((form.flags & FormatFlags::NO_WRITE) == 0) &&
        (form.conv_name != '%')) {
      listener << "\tpointer value: "
               << int_to_hex<uintptr_t>(
                      reinterpret_cast<uintptr_t>(form.output_ptr))
               << "\n";
    }

    listener << "\tconversion name: " << form.conv_name << "\n";

    if (form.conv_name == '[') {
      listener << "\t\t";
      for (size_t i = 0; i < 256 /* char max */; ++i) {
        if (form.scan_set.test(i)) {
          listener << static_cast<char>(i);
        }
      }
      listener << "\n\t]\n";
    }
  }
}
} // anonymous namespace

bool FormatSectionMatcher::MatchAndExplain(
    const FormatSection &actual, MatchResultListener *listener) const {
  if (expected == actual)
    return true;
  if (!listener->IsInterested())
    return false;
  *listener << "expected format section: ";
  display(*listener, expected);
  *listener << "\nactual format section  : ";
  display(*listener, actual);
  *listener << '\n';
  return false;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL
