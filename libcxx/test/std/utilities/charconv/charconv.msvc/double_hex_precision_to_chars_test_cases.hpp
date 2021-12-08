// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef DOUBLE_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP
#define DOUBLE_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP

#include <charconv>

#include "test.hpp"
using namespace std;

inline constexpr DoublePrecisionToCharsTestCase double_hex_precision_to_chars_test_cases[] = {
    // Test special cases (zero, inf, nan) and an ordinary case. Also test negative signs.
    {0.0, chars_format::hex, 4, "0.0000p+0"},
    {-0.0, chars_format::hex, 4, "-0.0000p+0"},
    {double_inf, chars_format::hex, 4, "inf"},
    {-double_inf, chars_format::hex, 4, "-inf"},
    {double_nan, chars_format::hex, 4, "nan"},
    {-double_nan, chars_format::hex, 4, "-nan(ind)"},
    {double_nan_payload, chars_format::hex, 4, "nan"},
    {-double_nan_payload, chars_format::hex, 4, "-nan"},
    {0x1.729p+0, chars_format::hex, 4, "1.7290p+0"},
    {-0x1.729p+0, chars_format::hex, 4, "-1.7290p+0"},

    // Test hexfloat corner cases.
    {0x1.728p+0, chars_format::hex, 13, "1.7280000000000p+0"},
    {0x0.0000000000001p-1022, chars_format::hex, 13, "0.0000000000001p-1022"}, // min subnormal
    {0x0.fffffffffffffp-1022, chars_format::hex, 13, "0.fffffffffffffp-1022"}, // max subnormal
    {0x1p-1022, chars_format::hex, 13, "1.0000000000000p-1022"}, // min normal
    {0x1.fffffffffffffp+1023, chars_format::hex, 13, "1.fffffffffffffp+1023"}, // max normal

    // Test hexfloat exponents.
    {0x1p-1009, chars_format::hex, 0, "1p-1009"},
    {0x1p-999, chars_format::hex, 0, "1p-999"},
    {0x1p-99, chars_format::hex, 0, "1p-99"},
    {0x1p-9, chars_format::hex, 0, "1p-9"},
    {0x1p+0, chars_format::hex, 0, "1p+0"},
    {0x1p+9, chars_format::hex, 0, "1p+9"},
    {0x1p+99, chars_format::hex, 0, "1p+99"},
    {0x1p+999, chars_format::hex, 0, "1p+999"},
    {0x1p+1009, chars_format::hex, 0, "1p+1009"},

    // Test hexfloat hexits.
    {0x1.01234567p+0, chars_format::hex, 8, "1.01234567p+0"},
    {0x1.89abcdefp+0, chars_format::hex, 8, "1.89abcdefp+0"},

    // Test varying precision. Negative precision requests full precision, not shortest round-trip.
    {0x1.1234561234561p+0, chars_format::hex, -2, "1.1234561234561p+0"},
    {0x1.1234561234561p+0, chars_format::hex, -1, "1.1234561234561p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 0, "1p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 1, "1.1p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 2, "1.12p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 3, "1.123p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 4, "1.1234p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 5, "1.12345p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 6, "1.123456p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 7, "1.1234561p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 8, "1.12345612p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 9, "1.123456123p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 10, "1.1234561234p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 11, "1.12345612345p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 12, "1.123456123456p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 13, "1.1234561234561p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 14, "1.12345612345610p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 15, "1.123456123456100p+0"},
    {0x1.1234561234561p+0, chars_format::hex, 16, "1.1234561234561000p+0"},

    // Test rounding at every position.
    {0x1.cccccccccccccp+0, chars_format::hex, 0, "2p+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 1, "1.dp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 2, "1.cdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 3, "1.ccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 4, "1.cccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 5, "1.ccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 6, "1.cccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 7, "1.ccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 8, "1.cccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 9, "1.ccccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 10, "1.cccccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 11, "1.ccccccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 12, "1.cccccccccccdp+0"},
    {0x1.cccccccccccccp+0, chars_format::hex, 13, "1.cccccccccccccp+0"},

    // Test all combinations of least significant bit, round bit, and trailing bits.
    {0x1.04000p+0, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0, Trailing 0
    {0x1.04001p+0, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0 and Trailing 1 in different hexits
    {0x1.04200p+0, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0 and Trailing 1 in same hexit
    {0x1.04800p+0, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 1, Trailing 0
    {0x1.04801p+0, chars_format::hex, 2, "1.05p+0"}, // Lsb 0, Round 1 and Trailing 1 in different hexits
    {0x1.04900p+0, chars_format::hex, 2, "1.05p+0"}, // Lsb 0, Round 1 and Trailing 1 in same hexit
    {0x1.05000p+0, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0, Trailing 0
    {0x1.05001p+0, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0 and Trailing 1 in different hexits
    {0x1.05200p+0, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0 and Trailing 1 in same hexit
    {0x1.05800p+0, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1, Trailing 0
    {0x1.05801p+0, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1 and Trailing 1 in different hexits
    {0x1.05900p+0, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1 and Trailing 1 in same hexit

    // Test carry propagation.
    {0x1.0affffffffffep+0, chars_format::hex, 12, "1.0b0000000000p+0"},

    // Test carry propagation into the leading hexit.
    {0x0.fffffffffffffp-1022, chars_format::hex, 12, "1.000000000000p-1022"},
    {0x1.fffffffffffffp+1023, chars_format::hex, 12, "2.000000000000p+1023"},

    // Test how the leading hexit participates in the rounding decision.
    {0x0.000p+0, chars_format::hex, 0, "0p+0"},
    {0x0.001p-1022, chars_format::hex, 0, "0p-1022"},
    {0x0.200p-1022, chars_format::hex, 0, "0p-1022"},
    {0x0.800p-1022, chars_format::hex, 0, "0p-1022"},
    {0x0.801p-1022, chars_format::hex, 0, "1p-1022"},
    {0x0.900p-1022, chars_format::hex, 0, "1p-1022"},
    {0x1.000p+0, chars_format::hex, 0, "1p+0"},
    {0x1.001p+0, chars_format::hex, 0, "1p+0"},
    {0x1.200p+0, chars_format::hex, 0, "1p+0"},
    {0x1.800p+0, chars_format::hex, 0, "2p+0"},
    {0x1.801p+0, chars_format::hex, 0, "2p+0"},
    {0x1.900p+0, chars_format::hex, 0, "2p+0"},
};

#endif // DOUBLE_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP
