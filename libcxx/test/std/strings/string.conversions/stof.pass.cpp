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

// float stof(const string& str, size_t *idx = 0);
// float stof(const wstring& str, size_t *idx = 0);

#include <string>
#include <cmath>
#include <cassert>

int main()
{
    assert(std::stof("0") == 0);
    assert(std::stof(L"0") == 0);
    assert(std::stof("-0") == 0);
    assert(std::stof(L"-0") == 0);
    assert(std::stof("-10") == -10);
    assert(std::stof(L"-10.5") == -10.5);
    assert(std::stof(" 10") == 10);
    assert(std::stof(L" 10") == 10);
    size_t idx = 0;
    assert(std::stof("10g", &idx) == 10);
    assert(idx == 2);
    idx = 0;
    assert(std::stof(L"10g", &idx) == 10);
    assert(idx == 2);
    idx = 0;
    try
    {
        assert(std::stof("1.e60", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        assert(std::stof(L"1.e60", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    idx = 0;
    try
    {
        assert(std::stof("1.e360", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        assert(std::stof(L"1.e360", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        assert(std::stof("INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::stof(L"INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::isnan(std::stof("NAN", &idx)));
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::isnan(std::stof(L"NAN", &idx)));
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        std::stof("", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stof(L"", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stof("  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stof(L"  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stof("a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stof(L"a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
}
