//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(wstring_convert&& other); // EXTENSION

#include <locale>
#include <codecvt>
#include <cassert>

int main()
{
    typedef std::codecvt_utf8<wchar_t> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
    // create a converter and perform some conversions to generate some
    // interesting state.
    Myconv myconv;
    myconv.from_bytes("\xF1\x80\x80\x83");
    const int old_converted = myconv.converted();
    assert(myconv.converted() == 4);
    // move construct a new converter and make sure the state is the same.
    Myconv myconv2(std::move(myconv));
    assert(myconv2.converted() == 4);
}
