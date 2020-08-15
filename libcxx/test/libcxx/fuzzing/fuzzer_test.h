// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_FUZZER_TEST_H
#define TEST_LIBCXX_FUZZER_TEST_H

#include <cstddef>
#include <cassert>

#include "../../../fuzzing/fuzzing.h"
#include "../../../fuzzing/fuzzing.cpp"

const char* TestCaseSetOne[] = {"", "s", "bac",
                            "bacasf",
                            "lkajseravea",
                            "adsfkajdsfjkas;lnc441324513,34535r34525234",
                            "b*c",
                            "ba?sf"
                            "lka*ea",
                            "adsf*kas;lnc441[0-9]1r34525234"};

using FuzzerFuncType = int(const uint8_t*, size_t);

template <size_t NumCases>
inline void RunFuzzingTest(FuzzerFuncType *to_test, const char* (&test_cases)[NumCases]) {
  for (const char* TC : test_cases) {
    const size_t size = std::strlen(TC);
    const uint8_t* data = (const uint8_t*)TC;
    int result = to_test(data, size);
    assert(result == 0);
  }
}

#define FUZZER_TEST(FuncName) \
int main() { \
  RunFuzzingTest(FuncName, TestCaseSetOne); \
} \
extern int require_semi

#endif // TEST_LIBCXX_FUZZER_TEST_H
