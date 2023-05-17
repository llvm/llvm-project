//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// int stoi(const string& str, size_t *idx = 0, int base = 10);
// int stoi(const wstring& str, size_t *idx = 0, int base = 10);

#include <string>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

int main(int, char**)
{
    assert(std::stoi("0") == 0);
    assert(std::stoi("-0") == 0);
    assert(std::stoi("-10") == -10);
    assert(std::stoi(" 10") == 10);
    {
        std::size_t idx = 0;
        assert(std::stoi("10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (std::numeric_limits<long>::max() > std::numeric_limits<int>::max()) {
        std::size_t idx = 0;
        try {
            (void)std::stoi("0x100000000", &idx, 16);
            assert(false);
        } catch (const std::out_of_range&) {

        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi("", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi("  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi("a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(std::stoi(L"0") == 0);
    assert(std::stoi(L"-0") == 0);
    assert(std::stoi(L"-10") == -10);
    assert(std::stoi(L" 10") == 10);
    {
        std::size_t idx = 0;
        assert(std::stoi(L"10g", &idx, 16) == 16);
        assert(idx == 2);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    if (std::numeric_limits<long>::max() > std::numeric_limits<int>::max()) {
        std::size_t idx = 0;
        try {
            (void)std::stoi(L"0x100000000", &idx, 16);
            assert(false);
        } catch (const std::out_of_range&) {

        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi(L"", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi(L"  - 8", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
    {
        std::size_t idx = 0;
        try {
            (void)std::stoi(L"a1", &idx);
            assert(false);
        } catch (const std::invalid_argument&) {
            assert(idx == 0);
        }
    }
#endif // TEST_HAS_NO_EXCEPTIONS
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
