//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// int sync();

#include <istream>
#include <cassert>

#include "test_macros.h"

int sync_called = 0;

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

protected:
    int sync()
    {
        ++sync_called;
        return 5;
    }
};

#ifndef TEST_HAS_NO_EXCEPTIONS
struct testbuf_exception { };

template <class CharT>
struct throwing_testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    throwing_testbuf() {}
    throwing_testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }

    CharT* eback() const {return base::eback();}
    CharT* gptr() const {return base::gptr();}
    CharT* egptr() const {return base::egptr();}

protected:
    virtual int sync()
    {
        throw testbuf_exception();
        return 5;
    }
};
#endif // TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
    {
        testbuf<char> sb(" 123456789");
        std::istream is(&sb);
        assert(is.sync() == 0);
        assert(sync_called == 1);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        testbuf<wchar_t> sb(L" 123456789");
        std::wistream is(&sb);
        assert(is.sync() == 0);
        assert(sync_called == 2);
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        throwing_testbuf<char> sb(" 123456789");
        std::basic_istream<char> is(&sb);
        is.exceptions(std::ios_base::badbit);
        bool threw = false;
        try {
            is.sync();
        } catch (testbuf_exception const&) {
            threw = true;
        }
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
        assert(threw);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        throwing_testbuf<wchar_t> sb(L" 123456789");
        std::basic_istream<wchar_t> is(&sb);
        is.exceptions(std::ios_base::badbit);
        bool threw = false;
        try {
            is.sync();
        } catch (testbuf_exception const&) {
            threw = true;
        }
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
        assert(threw);
    }
#endif
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
