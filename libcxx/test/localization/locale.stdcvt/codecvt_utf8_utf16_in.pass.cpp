//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <codecvt>

// template <class Elem, unsigned long Maxcode = 0x10ffff,
//           codecvt_mode Mode = (codecvt_mode)0>
// class codecvt_utf8_utf16
//     : public codecvt<Elem, char, mbstate_t>
// {
//     // unspecified
// };

// result
//     in(stateT& state,
//        const externT* from, const externT* from_end, const externT*& from_next,
//        internT* to, internT* to_end, internT*& to_next) const;

#include <codecvt>
#include <cassert>

int main()
{
    {
        typedef std::codecvt_utf8_utf16<wchar_t> C;
        C c;
        wchar_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        wchar_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+4);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<wchar_t, 0x1000> C;
        C c;
        wchar_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        wchar_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<wchar_t, 0x10ffff, std::consume_header> C;
        C c;
        wchar_t w[2] = {0};
        char n[7] = {char(0xEF), char(0xBB), char(0xBF), char(0xF1), char(0x80), char(0x80), char(0x83)};
        wchar_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+7, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+7);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char32_t> C;
        C c;
        char32_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        char32_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+4);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char32_t, 0x1000> C;
        C c;
        char32_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        char32_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char32_t, 0x10ffff, std::consume_header> C;
        C c;
        char32_t w[2] = {0};
        char n[7] = {char(0xEF), char(0xBB), char(0xBF), char(0xF1), char(0x80), char(0x80), char(0x83)};
        char32_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+7, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+7);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char16_t> C;
        C c;
        char16_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        char16_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+4);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char16_t, 0x1000> C;
        C c;
        char16_t w[2] = {0};
        char n[4] = {char(0xF1), char(0x80), char(0x80), char(0x83)};
        char16_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+4, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::error);
        assert(wp == w);
        assert(np == n);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
    {
        typedef std::codecvt_utf8_utf16<char16_t, 0x10ffff, std::consume_header> C;
        C c;
        char16_t w[2] = {0};
        char n[7] = {char(0xEF), char(0xBB), char(0xBF), char(0xF1), char(0x80), char(0x80), char(0x83)};
        char16_t* wp = nullptr;
        std::mbstate_t m;
        const char* np = nullptr;
        std::codecvt_base::result r = c.in(m, n, n+7, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+2);
        assert(np == n+7);
        assert(w[0] == 0xD8C0);
        assert(w[1] == 0xDC03);

        n[0] = char(0xE1);
        n[1] = char(0x80);
        n[2] = char(0x85);
        r = c.in(m, n, n+3, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+3);
        assert(w[0] == 0x1005);

        n[0] = char(0xD1);
        n[1] = char(0x93);
        r = c.in(m, n, n+2, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+2);
        assert(w[0] == 0x0453);

        n[0] = char(0x56);
        r = c.in(m, n, n+1, np, w, w+2, wp);
        assert(r == std::codecvt_base::ok);
        assert(wp == w+1);
        assert(np == n+1);
        assert(w[0] == 0x0056);
    }
}
