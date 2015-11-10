//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <string>

// double stod(const string& str, size_t *idx = 0);
// double stod(const wstring& str, size_t *idx = 0);

#include <string>
#include <cmath>
#include <cassert>

int main()
{
    assert(std::stod("0") == 0);
    assert(std::stod(L"0") == 0);
    assert(std::stod("-0") == 0);
    assert(std::stod(L"-0") == 0);
    assert(std::stod("-10") == -10);
    assert(std::stod(L"-10.5") == -10.5);
    assert(std::stod(" 10") == 10);
    assert(std::stod(L" 10") == 10);
    size_t idx = 0;
    assert(std::stod("10g", &idx) == 10);
    assert(idx == 2);
    idx = 0;
    assert(std::stod(L"10g", &idx) == 10);
    assert(idx == 2);
    try
    {
        assert(std::stod("1.e60", &idx) == 1.e60);
        assert(idx == 5);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    try
    {
        assert(std::stod(L"1.e60", &idx) == 1.e60);
        assert(idx == 5);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::stod("1.e360", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        assert(std::stod(L"1.e360", &idx) == INFINITY);
        assert(false);
    }
    catch (const std::out_of_range&)
    {
        assert(idx == 0);
    }
    try
    {
        assert(std::stod("INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::stod(L"INF", &idx) == INFINITY);
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::isnan(std::stod("NAN", &idx)));
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        assert(std::isnan(std::stod(L"NAN", &idx)));
        assert(idx == 3);
    }
    catch (const std::out_of_range&)
    {
        assert(false);
    }
    idx = 0;
    try
    {
        std::stod("", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stod(L"", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stod("  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stod(L"  - 8", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stod("a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
    try
    {
        std::stod(L"a1", &idx);
        assert(false);
    }
    catch (const std::invalid_argument&)
    {
        assert(idx == 0);
    }
}
