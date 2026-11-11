//===-- MemoryMatcher.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryMatcher.h"

#include "src/__support/ctype_utils.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

#if LIBC_TEST_HAS_MATCHERS()

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

static void display(MatchResultListener &listener, char C) {
  const auto print = [&listener](unsigned char i) {
    listener << LIBC_NAMESPACE::internal::toupper(
        LIBC_NAMESPACE::internal::int_to_b36_char(i));
  };
  print(static_cast<unsigned char>(C) / 16);
  print(static_cast<unsigned char>(C) & 15);
}

static void display(MatchResultListener &listener, MemoryView View) {
  for (auto C : View) {
    listener << ' ';
    display(listener, C);
  }
}

bool MemoryMatcher::MatchAndExplain(MemoryView actual,
                                    MatchResultListener *listener) const {
  bool mismatch_size = false;
  size_t mismatch_index = cpp::numeric_limits<size_t>::max();
  if (equals(expected, actual, mismatch_size, mismatch_index))
    return true;
  if (!listener->IsInterested())
    return false;
  if (mismatch_size) {
    *listener << "Size mismatch :expected : " << expected.size() << '\n'
              << "actual   : " << actual.size() << '\n';
  } else {
    *listener << "Mismatch at position : " << mismatch_index << " / "
              << expected.size() << "\nexpected :";
    display(*listener, expected);
    *listener << "\nactual   :";
    display(*listener, actual);
    *listener << '\n';
  }
  return false;
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TEST_HAS_MATCHERS()
