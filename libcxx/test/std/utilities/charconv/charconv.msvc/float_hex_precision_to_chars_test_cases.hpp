// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FLOAT_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP
#define FLOAT_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP

#include <charconv>

#include "test.hpp"
using namespace std;

inline constexpr FloatPrecisionToCharsTestCase float_hex_precision_to_chars_test_cases[] = {
    // Test special cases (zero, inf, nan) and an ordinary case. Also test negative signs.
    {0.0f, chars_format::hex, 4, "0.0000p+0"},
    {-0.0f, chars_format::hex, 4, "-0.0000p+0"},
    {float_inf, chars_format::hex, 4, "inf"},
    {-float_inf, chars_format::hex, 4, "-inf"},
    {float_nan, chars_format::hex, 4, "nan"},
    {-float_nan, chars_format::hex, 4, "-nan(ind)"},
    {float_nan_payload, chars_format::hex, 4, "nan"},
    {-float_nan_payload, chars_format::hex, 4, "-nan"},
    {0x1.729p+0f, chars_format::hex, 4, "1.7290p+0"},
    {-0x1.729p+0f, chars_format::hex, 4, "-1.7290p+0"},

    // Test hexfloat corner cases.
    {0x1.728p+0f, chars_format::hex, 6, "1.728000p+0"},
    {0x0.000002p-126f, chars_format::hex, 6, "0.000002p-126"}, // min subnormal
    {0x0.fffffep-126f, chars_format::hex, 6, "0.fffffep-126"}, // max subnormal
    {0x1p-126f, chars_format::hex, 6, "1.000000p-126"}, // min normal
    {0x1.fffffep+127f, chars_format::hex, 6, "1.fffffep+127"}, // max normal

    // Test hexfloat exponents.
    {0x1p-109f, chars_format::hex, 0, "1p-109"},
    {0x1p-99f, chars_format::hex, 0, "1p-99"},
    {0x1p-9f, chars_format::hex, 0, "1p-9"},
    {0x1p+0f, chars_format::hex, 0, "1p+0"},
    {0x1p+9f, chars_format::hex, 0, "1p+9"},
    {0x1p+99f, chars_format::hex, 0, "1p+99"},
    {0x1p+109f, chars_format::hex, 0, "1p+109"},

    // Test hexfloat hexits.
    {0x1.0123p+0f, chars_format::hex, 4, "1.0123p+0"},
    {0x1.4567p+0f, chars_format::hex, 4, "1.4567p+0"},
    {0x1.89abp+0f, chars_format::hex, 4, "1.89abp+0"},
    {0x1.cdefp+0f, chars_format::hex, 4, "1.cdefp+0"},

    // Test varying precision. Negative precision requests full precision, not shortest round-trip.
    {0x1.123456p+0f, chars_format::hex, -2, "1.123456p+0"},
    {0x1.123456p+0f, chars_format::hex, -1, "1.123456p+0"},
    {0x1.123456p+0f, chars_format::hex, 0, "1p+0"},
    {0x1.123456p+0f, chars_format::hex, 1, "1.1p+0"},
    {0x1.123456p+0f, chars_format::hex, 2, "1.12p+0"},
    {0x1.123456p+0f, chars_format::hex, 3, "1.123p+0"},
    {0x1.123456p+0f, chars_format::hex, 4, "1.1234p+0"},
    {0x1.123456p+0f, chars_format::hex, 5, "1.12345p+0"},
    {0x1.123456p+0f, chars_format::hex, 6, "1.123456p+0"},
    {0x1.123456p+0f, chars_format::hex, 7, "1.1234560p+0"},
    {0x1.123456p+0f, chars_format::hex, 8, "1.12345600p+0"},
    {0x1.123456p+0f, chars_format::hex, 9, "1.123456000p+0"},

    // Test rounding at every position.
    {0x1.ccccccp+0f, chars_format::hex, 0, "2p+0"},
    {0x1.ccccccp+0f, chars_format::hex, 1, "1.dp+0"},
    {0x1.ccccccp+0f, chars_format::hex, 2, "1.cdp+0"},
    {0x1.ccccccp+0f, chars_format::hex, 3, "1.ccdp+0"},
    {0x1.ccccccp+0f, chars_format::hex, 4, "1.cccdp+0"},
    {0x1.ccccccp+0f, chars_format::hex, 5, "1.ccccdp+0"},
    {0x1.ccccccp+0f, chars_format::hex, 6, "1.ccccccp+0"},

    // Test all combinations of least significant bit, round bit, and trailing bits.
    {0x1.04000p+0f, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0, Trailing 0
    {0x1.04001p+0f, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0 and Trailing 1 in different hexits
    {0x1.04200p+0f, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 0 and Trailing 1 in same hexit
    {0x1.04800p+0f, chars_format::hex, 2, "1.04p+0"}, // Lsb 0, Round 1, Trailing 0
    {0x1.04801p+0f, chars_format::hex, 2, "1.05p+0"}, // Lsb 0, Round 1 and Trailing 1 in different hexits
    {0x1.04900p+0f, chars_format::hex, 2, "1.05p+0"}, // Lsb 0, Round 1 and Trailing 1 in same hexit
    {0x1.05000p+0f, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0, Trailing 0
    {0x1.05001p+0f, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0 and Trailing 1 in different hexits
    {0x1.05200p+0f, chars_format::hex, 2, "1.05p+0"}, // Lsb 1, Round 0 and Trailing 1 in same hexit
    {0x1.05800p+0f, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1, Trailing 0
    {0x1.05801p+0f, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1 and Trailing 1 in different hexits
    {0x1.05900p+0f, chars_format::hex, 2, "1.06p+0"}, // Lsb 1, Round 1 and Trailing 1 in same hexit

    // Test carry propagation.
    {0x1.0afffep+0f, chars_format::hex, 5, "1.0b000p+0"},

    // Test carry propagation into the leading hexit.
    {0x0.fffffep-126f, chars_format::hex, 5, "1.00000p-126"},
    {0x1.fffffep+127f, chars_format::hex, 5, "2.00000p+127"},

    // Test how the leading hexit participates in the rounding decision.
    {0x0.000p+0f, chars_format::hex, 0, "0p+0"},
    {0x0.001p-126f, chars_format::hex, 0, "0p-126"},
    {0x0.200p-126f, chars_format::hex, 0, "0p-126"},
    {0x0.800p-126f, chars_format::hex, 0, "0p-126"},
    {0x0.801p-126f, chars_format::hex, 0, "1p-126"},
    {0x0.900p-126f, chars_format::hex, 0, "1p-126"},
    {0x1.000p+0f, chars_format::hex, 0, "1p+0"},
    {0x1.001p+0f, chars_format::hex, 0, "1p+0"},
    {0x1.200p+0f, chars_format::hex, 0, "1p+0"},
    {0x1.800p+0f, chars_format::hex, 0, "2p+0"},
    {0x1.801p+0f, chars_format::hex, 0, "2p+0"},
    {0x1.900p+0f, chars_format::hex, 0, "2p+0"},
};

#endif // FLOAT_HEX_PRECISION_TO_CHARS_TEST_CASES_HPP
