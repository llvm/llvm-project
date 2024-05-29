//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <istream>

// int sync();

// The fix for bug 51497 and bug 51499 require and updated dylib due to
// explicit instantiations. That means Apple backdeployment targets remain
// broken.
// TODO(#82107) Enable XFAIL.
// UNSUPPORTED: using-built-library-before-llvm-19

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

template <class CharT>
struct testbuf_pubsync_error
    : public std::basic_streambuf<CharT>
{
public:

    testbuf_pubsync_error() {}
protected:
    virtual int sync() { return -1; }
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
        std::istream is(nullptr);
        assert(is.sync() == -1);
    }
    {
        testbuf<char> sb(" 123456789");
        std::istream is(&sb);
        assert(is.sync() == 0);
        assert(sync_called == 1);
    }
    {
        testbuf_pubsync_error<char> sb;
        std::istream is(&sb);
        is.exceptions(std::ios_base::failbit | std::ios_base::eofbit);
        assert(is.sync() == -1);
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wistream is(nullptr);
        assert(is.sync() == -1);
    }
    {
        testbuf<wchar_t> sb(L" 123456789");
        std::wistream is(&sb);
        assert(is.sync() == 0);
        assert(sync_called == 2);
    }
    {
        testbuf_pubsync_error<wchar_t> sb;
        std::wistream is(&sb);
        is.exceptions(std::ios_base::failbit | std::ios_base::eofbit);
        assert(is.sync() == -1);
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
    }
#endif
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        testbuf_pubsync_error<char> sb;
        std::istream is(&sb);
        is.exceptions(std::ios_base::badbit);
        bool threw = false;
        try {
            is.sync();
        } catch (std::ios_base::failure const&) {
            threw = true;
        }
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
        assert(threw);
    }
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
        testbuf_pubsync_error<wchar_t> sb;
        std::wistream is(&sb);
        is.exceptions(std::ios_base::badbit);
        bool threw = false;
        try {
            is.sync();
        } catch (std::ios_base::failure const&) {
            threw = true;
        }
        assert( is.bad());
        assert(!is.eof());
        assert( is.fail());
        assert(threw);
    }
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
#endif // TEST_HAS_NO_WIDE_CHARACTERS
#endif // TEST_HAS_NO_EXCEPTIONS

    return 0;
}
