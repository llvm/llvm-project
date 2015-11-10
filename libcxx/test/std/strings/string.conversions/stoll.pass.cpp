//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XFAIL: libcpp-no-exceptions
// XFAIL: with_system_cxx_lib=x86_64-apple-darwin11
// XFAIL: with_system_cxx_lib=x86_64-apple-darwin12

// <string>

// long long stoll(const string& str, size_t *idx = 0, int base = 10);
// long long stoll(const wstring& str, size_t *idx = 0, int base = 10);

#include <string>
#include <cassert>

int main()
{
    assert(std::stoll("0") == 0);
    assert(std::stoll(L"0") == 0);
    assert(std::stoll("-0") == 0);
    assert(std::stoll(L"-0") == 0);
    assert(std::stoll("-10") == -10);
    assert(std::stoll(L"-10") == -10);
    assert(std::stoll(" 10") == 10);
    assert(std::stoll(L" 10") == 10);
    size_t idx = 0;
    assert(std::stoll("10g", &idx, 16) == 16);
    assert(idx == 2);
    idx = 0;
    assert(std::stoll(L"10g", &idx, 16) == 16);
    assert(idx == 2);
    idx = 0;
    try
    {
        std::stoll("", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll(L"", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll("  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll(L"  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll("a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll(L"a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll("99999999999999999999999999", &idx);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stoll(L"99999999999999999999999999", &idx);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
}
