//===-- Unittests for IntegerSequence -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Utility.h"
#include "utils/UnitTest/Test.h"

using namespace __llvm_libc::cpp;

TEST(LlvmLibcIntegerSequencetTest, Basic) {
  EXPECT_TRUE((is_same_v<IntegerSequence<int>, MakeIntegerSequence<int, 0>>));
  using ISeq = IntegerSequence<int, 0, 1, 2, 3>;
  EXPECT_TRUE((is_same_v<ISeq, MakeIntegerSequence<int, 4>>));
  using LSeq = IntegerSequence<long, 0, 1, 2, 3>;
  EXPECT_TRUE((is_same_v<LSeq, MakeIntegerSequence<long, 4>>));
  using ULLSeq = IntegerSequence<unsigned long long, 0ull, 1ull, 2ull, 3ull>;
  EXPECT_TRUE((is_same_v<ULLSeq, MakeIntegerSequence<unsigned long long, 4>>));
}

template <typename T, T... Ts>
bool checkArray(IntegerSequence<T, Ts...> seq) {
  T arr[sizeof...(Ts)]{Ts...};

  for (T i = 0; i < static_cast<T>(sizeof...(Ts)); i++)
    if (arr[i] != i)
      return false;

  return true;
}

TEST(LlvmLibcIntegerSequencetTest, Many) {
  EXPECT_TRUE(checkArray(MakeIntegerSequence<int, 100>{}));
}
