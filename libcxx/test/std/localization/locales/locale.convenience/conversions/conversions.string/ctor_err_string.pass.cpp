//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(const byte_string& byte_err,
//                 const wide_string& wide_err = wide_string());

#include <locale>
#include <codecvt>
#include <cassert>

int main()
{
    typedef std::codecvt_utf8<wchar_t> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
#if _LIBCPP_STD_VER > 11
    static_assert(!std::is_convertible<std::string, Myconv>::value, "");
    static_assert( std::is_constructible<Myconv, std::string>::value, "");
#endif
    {
        Myconv myconv;
        try
        {
            myconv.to_bytes(L"\xDA83");
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
        try
        {
            myconv.from_bytes('\xA5');
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
    }
    {
        Myconv myconv("byte error");
        std::string bs = myconv.to_bytes(L"\xDA83");
        assert(bs == "byte error");
        try
        {
            myconv.from_bytes('\xA5');
            assert(false);
        }
        catch (const std::range_error&)
        {
        }
    }
    {
        Myconv myconv("byte error", L"wide error");
        std::string bs = myconv.to_bytes(L"\xDA83");
        assert(bs == "byte error");
        std::wstring ws = myconv.from_bytes('\xA5');
        assert(ws == L"wide error");
    }
}
