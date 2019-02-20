//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// template <class charT, class traits>
//   basic_istream<charT,traits>&
//   ws(basic_istream<charT,traits>& is);

#include <istream>
#include <cassert>

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    testbuf() {}
    testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }

    CharT* eback() const {return base::eback();}
    CharT* gptr() const {return base::gptr();}
    CharT* egptr() const {return base::egptr();}
};

int main(int, char**)
{
    {
        testbuf<char> sb("   123");
        std::istream is(&sb);
        ws(is);
        assert(is.good());
        assert(is.peek() == '1');
    }
    {
        testbuf<wchar_t> sb(L"   123");
        std::wistream is(&sb);
        ws(is);
        assert(is.good());
        assert(is.peek() == L'1');
    }
    {
        testbuf<char> sb("  ");
        std::istream is(&sb);
        ws(is);
        assert(!is.fail());
        assert(is.eof());
        ws(is);
        assert(is.eof());
        assert(is.fail());
    }
    {
        testbuf<wchar_t> sb(L"  ");
        std::wistream is(&sb);
        ws(is);
        assert(!is.fail());
        assert(is.eof());
        ws(is);
        assert(is.eof());
        assert(is.fail());
    }

  return 0;
}
