//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// long double stold(const string& str, size_t *idx = 0);
// long double stold(const wstring& str, size_t *idx = 0);

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::stold("0") == 0);
    assert(std::stold("-0") == 0);
    assert(std::stold("-10") == -10);
    assert(std::stold(" 10") == 10);
    {
        std::size_t idx = 0;
        assert(std::stold("10g", &idx) == 10);
        assert(idx == 2);
    }
    {
        std::size_t idx = 0;
        assert(std::stold("1.e60", &idx) == 1.e60L);
        assert(idx == 5);
    }
    {
        std::size_t idx = 0;
        assert(std::stold("INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    {
        std::size_t idx = 0;
        assert(std::isnan(std::stold("NAN", &idx)));
        assert(idx == 3);
    }

#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::size_t idx = 0;
        try {
            (void)std::stold("", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stold("  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stold("a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            assert(std::stold("1.e6000", &idx) == INFINITY);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(std::stold(L"0") == 0);
    assert(std::stold(L"-0") == 0);
    assert(std::stold(L"-10.5") == -10.5);
    assert(std::stold(L" 10") == 10);
    {
        std::size_t idx = 0;
        assert(std::stold(L"10g", &idx) == 10);
        assert(idx == 2);
    }
    {
        std::size_t idx = 0;
        assert(std::stold(L"1.e60", &idx) == 1.e60L);
        assert(idx == 5);
    }
    {
        std::size_t idx = 0;
        assert(std::stold(L"INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    {
        std::size_t idx = 0;
        assert(std::isnan(std::stold(L"NAN", &idx)));
        assert(idx == 3);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::size_t idx = 0;
        try {
            (void)std::stold(L"", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stold(L"  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stold(L"a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            assert(std::stold(L"1.e6000", &idx) == INFINITY);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
