//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <istream>

// template <class Stream, class T>
// Stream&& operator>>(Stream&& is, T&& x);

#include <istream>
#include <sstream>
#include <cassert>

#include "test_macros.h"

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

struct Int {
    int value;
    template <class CharT>
    friend void operator>>(std::basic_istream<CharT>& is, Int& self) {
        is >> self.value;
    }
};

struct A { };
bool called = false;
void operator>>(std::istream&, A&&) { called = true; }

int main(int, char**)
{
    {
        testbuf<char> sb("   123");
        Int i = {0};
        std::istream is(&sb);
        std::istream&& result = (std::move(is) >> i);
        assert(&result == &is);
        assert(i.value == 123);
    }
    {
        testbuf<wchar_t> sb(L"   123");
        Int i = {0};
        std::wistream is(&sb);
        std::wistream&& result = (std::move(is) >> i);
        assert(&result == &is);
        assert(i.value == 123);
    }
    {
        // test perfect forwarding
        assert(called == false);
        std::istringstream ss;
        std::istringstream&& result = (std::move(ss) >> A{});
        assert(&result == &ss);
        assert(called);
    }

    return 0;
}
