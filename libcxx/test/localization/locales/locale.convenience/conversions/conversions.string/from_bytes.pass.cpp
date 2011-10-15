//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wide_string from_bytes(char byte);
// wide_string from_bytes(const char* ptr);
// wide_string from_bytes(const byte_string& str);
// wide_string from_bytes(const char* first, const char* last);

#include <locale>
#include <codecvt>
#include <cassert>

int main()
{
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t> > myconv;
        std::string bs("\xF1\x80\x80\x83");
        std::wstring ws = myconv.from_bytes('a');
        assert(ws == L"a");
        ws = myconv.from_bytes(bs.c_str());
        assert(ws == L"\x40003");
        ws = myconv.from_bytes(bs);
        assert(ws == L"\x40003");
        ws = myconv.from_bytes(bs.data(), bs.data() + bs.size());
        assert(ws == L"\x40003");
    }
}
