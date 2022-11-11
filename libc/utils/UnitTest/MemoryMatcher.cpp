//===-- MemoryMatcher.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MemoryMatcher.h"

namespace __llvm_libc {
namespace memory {
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

void display(testutils::StreamWrapper &Stream, char C) {
  const auto print = [&Stream](unsigned char I) {
    Stream << static_cast<char>(I < 10 ? '0' + I : 'A' + I - 10);
  };
  print(static_cast<unsigned char>(C) / 16);
  print(static_cast<unsigned char>(C) & 15);
}

void display(testutils::StreamWrapper &Stream, MemoryView View) {
  for (auto C : View) {
    Stream << ' ';
    display(Stream, C);
  }
}

void MemoryMatcher::explainError(testutils::StreamWrapper &Stream) {
  if (mismatch_size) {
    Stream << "Size mismatch :";
    Stream << "expected : ";
    Stream << expected.size();
    Stream << '\n';
    Stream << "actual   : ";
    Stream << actual.size();
    Stream << '\n';
  } else {
    Stream << "Mismatch at position : ";
    Stream << mismatch_index;
    Stream << " / ";
    Stream << expected.size();
    Stream << "\n";
    Stream << "expected :";
    display(Stream, expected);
    Stream << '\n';
    Stream << "actual   :";
    display(Stream, actual);
    Stream << '\n';
  }
}

} // namespace testing
} // namespace memory
} // namespace __llvm_libc
