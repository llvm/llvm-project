// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TEST_HPP
#define TEST_HPP

#include <charconv>
#include <cstddef>
#include <limits>
#include <system_error>
using namespace std;

inline constexpr float float_inf         = numeric_limits<float>::infinity();
inline constexpr float float_nan         = numeric_limits<float>::quiet_NaN();
inline constexpr float float_nan_payload = __builtin_nanf("1729");

inline constexpr double double_inf         = numeric_limits<double>::infinity();
inline constexpr double double_nan         = numeric_limits<double>::quiet_NaN();
inline constexpr double double_nan_payload = __builtin_nan("1729");

struct FloatFromCharsTestCase {
    const char* input;
    chars_format fmt;
    std::size_t correct_idx;
    errc correct_ec;
    float correct_value;
};

struct FloatToCharsTestCase {
    float value;
    chars_format fmt;
    const char* correct;
};

struct FloatPrecisionToCharsTestCase {
    float value;
    chars_format fmt;
    int precision;
    const char* correct;
};

struct DoubleFromCharsTestCase {
    const char* input;
    chars_format fmt;
    std::size_t correct_idx;
    errc correct_ec;
    double correct_value;
};

struct DoubleToCharsTestCase {
    double value;
    chars_format fmt;
    const char* correct;
};

struct DoublePrecisionToCharsTestCase {
    double value;
    chars_format fmt;
    int precision;
    const char* correct;
};

#endif // TEST_HPP
