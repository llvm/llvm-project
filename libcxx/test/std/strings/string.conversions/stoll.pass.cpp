//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// long long stoll(const string& str, size_t *idx = 0, int base = 10);
// long long stoll(const wstring& str, size_t *idx = 0, int base = 10);

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::stoll("0") == 0);
    assert(std::stoll("-0") == 0);
    assert(std::stoll("-10") == -10);
    assert(std::stoll(" 10") == 10);
    {
        size_t idx = 0;
        assert(std::stoll("10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        size_t idx = 0;
        try {
            (void)std::stoll("", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoll("  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoll("a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            // LWG#2009 and PR14919
            (void)std::stoll("99999999999999999999999999", &idx);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(std::stoll(L"0") == 0);
    assert(std::stoll(L"-0") == 0);
    assert(std::stoll(L"-10") == -10);
    assert(std::stoll(L" 10") == 10);
    {
        size_t idx = 0;
        assert(std::stoll(L"10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        size_t idx = 0;
        try {
            (void)std::stoll(L"", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoll(L"  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoll(L"a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            // LWG#2009 and PR14919
            (void)std::stoll(L"99999999999999999999999999", &idx);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
