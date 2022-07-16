//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// unsigned long long stoull(const string& str, size_t *idx = 0, int base = 10);
// unsigned long long stoull(const wstring& str, size_t *idx = 0, int base = 10);

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::stoull("0") == 0);
    assert(std::stoull("-0") == 0);
    assert(std::stoull(" 10") == 10);
    {
        size_t idx = 0;
        assert(std::stoull("10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        size_t idx = 0;
        try {
            (void)std::stoull("", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoull("  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoull("a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            // LWG#2009 and PR14919
            (void)std::stoull("9999999999999999999999999999999999999999999999999", &idx);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(std::stoull(L"0") == 0);
    assert(std::stoull(L"-0") == 0);
    assert(std::stoull(L" 10") == 10);
    {
        size_t idx = 0;
        assert(std::stoull(L"10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        size_t idx = 0;
        try {
            (void)std::stoull(L"", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoull(L"  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            (void)std::stoull(L"a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        size_t idx = 0;
        try {
            // LWG#2009 and PR14919
            (void)std::stoull(L"9999999999999999999999999999999999999999999999999", &idx);
            assert(false);
        } catch (const std::out_of_range&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
