//===-- ScanfMatcher.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScanfMatcher.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "src/stdio/scanf_core/core_structs.h"

#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace testing {

using scanf_core::FormatFlags;
using scanf_core::FormatSection;
using scanf_core::LengthModifier;

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

void display(FormatSection form) {
  tlog << "Raw String (len " << form.raw_string.size() << "): \"";
  for (size_t i = 0; i < form.raw_string.size(); ++i) {
    tlog << form.raw_string[i];
  }
  tlog << "\"";
  if (form.has_conv) {
    tlog << "\n\tHas Conv\n\tFlags:";
    IF_FLAG_SHOW_FLAG(NO_WRITE);
    IF_FLAG_SHOW_FLAG(ALLOCATE);
    tlog << "\n";
    tlog << "\tmax width: " << form.max_width << "\n";
    tlog << "\tlength modifier: ";
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
    tlog << "\n";
    // If the pointer is used (NO_WRITE is not set and the conversion isn't %).
    if (((form.flags & FormatFlags::NO_WRITE) == 0) &&
        (form.conv_name != '%')) {
      tlog << "\tpointer value: "
           << int_to_hex<uintptr_t>(
                  reinterpret_cast<uintptr_t>(form.output_ptr))
           << "\n";
    }

    tlog << "\tconversion name: " << form.conv_name << "\n";

    if (form.conv_name == '[') {
      tlog << "\t\t";
      for (size_t i = 0; i < 256 /* char max */; ++i) {
        if (form.scan_set.test(i)) {
          tlog << static_cast<char>(i);
        }
      }
      tlog << "\n\t]\n";
    }
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
} // namespace LIBC_NAMESPACE_DECL
