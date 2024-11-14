//===-- MemoryMatcher.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryMatcher.h"

#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

#if LIBC_TEST_HAS_MATCHERS()

using LIBC_NAMESPACE::testing::tlog;

namespace LIBC_NAMESPACE_DECL {
namespace testing {

template <typename T>
bool equals(const cpp::span<T> &Span1, const cpp::span<T> &Span2,
            bool &mismatch_size, size_t &mismatch_index) {
  if (Span1.size() != Span2.size()) {
    mismatch_size = true;
    return false;
  }
  for (size_t Index = 0; Index < Span1.size(); ++Index)
    if (Span1[Index] != Span2[Index]) {
      mismatch_index = Index;
      return false;
    }
  return true;
}

bool MemoryMatcher::match(MemoryView actualValue) {
  actual = actualValue;
  return equals(expected, actual, mismatch_size, mismatch_index);
}

static void display(char C) {
  const auto print = [](unsigned char I) {
    tlog << static_cast<char>(I < 10 ? '0' + I : 'A' + I - 10);
  };
  print(static_cast<unsigned char>(C) / 16);
  print(static_cast<unsigned char>(C) & 15);
}

static void display(MemoryView View) {
  for (auto C : View) {
    tlog << ' ';
    display(C);
  }
}

void MemoryMatcher::explainError() {
  if (mismatch_size) {
    tlog << "Size mismatch :";
    tlog << "expected : ";
    tlog << expected.size();
    tlog << '\n';
    tlog << "actual   : ";
    tlog << actual.size();
    tlog << '\n';
  } else {
    tlog << "Mismatch at position : ";
    tlog << mismatch_index;
    tlog << " / ";
    tlog << expected.size();
    tlog << "\n";
    tlog << "expected :";
    display(expected);
    tlog << '\n';
    tlog << "actual   :";
    display(actual);
    tlog << '\n';
  }
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TEST_HAS_MATCHERS()
