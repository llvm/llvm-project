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

// byte_string to_bytes(Elem wchar);
// byte_string to_bytes(const Elem* wptr);
// byte_string to_bytes(const wide_string& wstr);
// byte_string to_bytes(const Elem* first, const Elem* last);

#include <locale>
#include <codecvt>
#include <cassert>

int main()
{
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t> > myconv;
        std::wstring ws(1, L'\x40003');
        std::string bs = myconv.to_bytes(ws[0]);
        assert(bs == "\xF1\x80\x80\x83");
        bs = myconv.to_bytes(ws.c_str());
        assert(bs == "\xF1\x80\x80\x83");
        bs = myconv.to_bytes(ws);
        assert(bs == "\xF1\x80\x80\x83");
        bs = myconv.to_bytes(ws.data(), ws.data() + ws.size());
        assert(bs == "\xF1\x80\x80\x83");
    }
}
