//===-- ScanfMatcher.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScanfMatcher.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/stdio/scanf_core/core_structs.h"

#include "test/UnitTest/StringUtils.h"

#include <stdint.h>

namespace __llvm_libc {
namespace scanf_core {
namespace testing {

bool FormatSectionMatcher::match(FormatSection actualValue) {
  actual = actualValue;
  return expected == actual;
}

namespace {

#define IF_FLAG_SHOW_FLAG(flag_name)                                           \
  do {                                                                         \
    if ((form.flags & FormatFlags::flag_name) == FormatFlags::flag_name)       \
      stream << "\n\t\t" << #flag_name;                                        \
  } while (false)
#define CASE_LM(lm)                                                            \
  case (LengthModifier::lm):                                                   \
    stream << #lm;                                                             \
    break

void display(testutils::StreamWrapper &stream, FormatSection form) {
  stream << "Raw String (len " << form.raw_string.size() << "): \"";
  for (size_t i = 0; i < form.raw_string.size(); ++i) {
    stream << form.raw_string[i];
  }
  stream << "\"";
  if (form.has_conv) {
    stream << "\n\tHas Conv\n\tFlags:";
    IF_FLAG_SHOW_FLAG(NO_WRITE);
    IF_FLAG_SHOW_FLAG(ALLOCATE);
    stream << "\n";
    stream << "\tmax width: " << form.max_width << "\n";
    stream << "\tlength modifier: ";
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
    stream << "\n";
    // If the pointer is used (NO_WRITE is not set and the conversion isn't %).
    if (((form.flags & FormatFlags::NO_WRITE) == 0) &&
        (form.conv_name != '%')) {
      stream << "\tpointer value: "
             << int_to_hex<uintptr_t>(
                    reinterpret_cast<uintptr_t>(form.output_ptr))
             << "\n";
    }

    stream << "\tconversion name: " << form.conv_name << "\n";

    if (form.conv_name == '[') {
      stream << "\t\t";
      for (size_t i = 0; i < 256 /* char max */; ++i) {
        if (form.scan_set.test(i)) {
          stream << static_cast<char>(i);
        }
      }
      stream << "\n\t]\n";
    }
  }
}
} // anonymous namespace

void FormatSectionMatcher::explainError(testutils::StreamWrapper &stream) {
  stream << "expected format section: ";
  display(stream, expected);
  stream << '\n';
  stream << "actual format section  : ";
  display(stream, actual);
  stream << '\n';
}

} // namespace testing
} // namespace scanf_core
} // namespace __llvm_libc
