//===------------------------ strstream.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "strstream"
#include "algorithm"
#include "climits"
#include "cstring"

_LIBCPP_BEGIN_NAMESPACE_STD

strstreambuf::strstreambuf(streamsize __alsize)
    : __strmode_(__dynamic),
      __alsize_(__alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
}

strstreambuf::strstreambuf(void* (*__palloc)(size_t), void (*__pfree)(void*))
    : __strmode_(__dynamic),
      __alsize_(__default_alsize),
      __palloc_(__palloc),
      __pfree_(__pfree)
{
}

void
strstreambuf::__init(char* __gnext, streamsize __n, char* __pbeg)
{
    if (__n == 0)
        __n = strlen(__gnext);
    else if (__n < 0)
        __n = INT_MAX;
    if (__pbeg == nullptr)
        setg(__gnext, __gnext, __gnext + __n);
    else
    {
        setg(__gnext, __gnext, __pbeg);
        setp(__pbeg, __pbeg + __n);
    }
}

strstreambuf::strstreambuf(char* __gnext, streamsize __n, char* __pbeg)
    : __strmode_(),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init(__gnext, __n, __pbeg);
}

strstreambuf::strstreambuf(const char* __gnext, streamsize __n)
    : __strmode_(__constant),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init((char*)__gnext, __n, nullptr);
}

strstreambuf::strstreambuf(signed char* __gnext, streamsize __n, signed char* __pbeg)
    : __strmode_(),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init((char*)__gnext, __n, (char*)__pbeg);
}

strstreambuf::strstreambuf(const signed char* __gnext, streamsize __n)
    : __strmode_(__constant),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init((char*)__gnext, __n, nullptr);
}

strstreambuf::strstreambuf(unsigned char* __gnext, streamsize __n, unsigned char* __pbeg)
    : __strmode_(),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init((char*)__gnext, __n, (char*)__pbeg);
}

strstreambuf::strstreambuf(const unsigned char* __gnext, streamsize __n)
    : __strmode_(__constant),
      __alsize_(__default_alsize),
      __palloc_(nullptr),
      __pfree_(nullptr)
{
    __init((char*)__gnext, __n, nullptr);
}

strstreambuf::~strstreambuf()
{
    if (eback() && (__strmode_ & __allocated) != 0 && (__strmode_ & __frozen) == 0)
    {
        if (__pfree_)
            __pfree_(eback());
        else
            delete [] eback();
    }
}

void
strstreambuf::swap(strstreambuf& __rhs)
{
    streambuf::swap(__rhs);
    _VSTD::swap(__strmode_, __rhs.__strmode_);
    _VSTD::swap(__alsize_, __rhs.__alsize_);
    _VSTD::swap(__palloc_, __rhs.__palloc_);
    _VSTD::swap(__pfree_, __rhs.__pfree_);
}

void
strstreambuf::freeze(bool __freezefl)
{
    if (__strmode_ & __dynamic)
    {
        if (__freezefl)
            __strmode_ |= __frozen;
        else
            __strmode_ &= ~__frozen;
    }
}

char*
strstreambuf::str()
{
    if (__strmode_ & __dynamic)
        __strmode_ |= __frozen;
    return eback();
}

int
strstreambuf::pcount() const
{
    return static_cast<int>(pptr() - pbase());
}

strstreambuf::int_type
strstreambuf::overflow(int_type __c)
{
    if (__c == EOF)
        return int_type(0);
    if (pptr() == epptr())
    {
        if ((__strmode_ & __dynamic) == 0 || (__strmode_ & __frozen) != 0)
            return int_type(EOF);
        streamsize old_size = (epptr() ? epptr() : egptr()) - eback();
        streamsize new_size = max<streamsize>(__alsize_, 2*old_size);
        char* buf = nullptr;
        if (__palloc_)
            buf = static_cast<char*>(__palloc_(new_size));
        else
            buf = new char[new_size];
        if (buf == nullptr)
            return int_type(EOF);
        memcpy(buf, eback(), old_size);
        ptrdiff_t ninp = gptr()  - eback();
        ptrdiff_t einp = egptr() - eback();
        ptrdiff_t nout = pptr()  - pbase();
        ptrdiff_t eout = epptr() - pbase();
        if (__strmode_ & __allocated)
        {
            if (__pfree_)
                __pfree_(eback());
            else
                delete [] eback();
        }
        setg(buf, buf + ninp, buf + einp);
        setp(buf + einp, buf + einp + eout);
        pbump(nout);
        __strmode_ |= __allocated;
    }
    *pptr() = static_cast<char>(__c);
    pbump(1);
    return int_type((unsigned char)__c);
}

strstreambuf::int_type
strstreambuf::pbackfail(int_type __c)
{
    if (eback() == gptr())
        return EOF;
    if (__c == EOF)
    {
        gbump(-1);
        return int_type(0);
    }
    if (__strmode_ & __constant)
    {
        if (gptr()[-1] == static_cast<char>(__c))
        {
            gbump(-1);
            return __c;
        }
        return EOF;
    }
    gbump(-1);
    *gptr() = static_cast<char>(__c);
    return __c;
}

strstreambuf::int_type
strstreambuf::underflow()
{
    if (gptr() == egptr())
    {
        if (egptr() >= pptr())
            return EOF;
        setg(eback(), gptr(), pptr());
    }
    return int_type((unsigned char)*gptr());
}

strstreambuf::pos_type
strstreambuf::seekoff(off_type __off, ios_base::seekdir __way, ios_base::openmode __which)
{
    off_type __p(-1);
    bool pos_in = __which & ios::in;
    bool pos_out = __which & ios::out;
    bool legal = false;
    switch (__way)
    {
    case ios::beg:
    case ios::end:
        if (pos_in || pos_out)
            legal = true;
        break;
    case ios::cur:
        if (pos_in != pos_out)
            legal = true;
        break;
    }
    if (pos_in && gptr() == nullptr)
        legal = false;
    if (pos_out && pptr() == nullptr)
        legal = false;
    if (legal)
    {
        off_type newoff;
        char* seekhigh = epptr() ? epptr() : egptr();
        switch (__way)
        {
        case ios::beg:
            newoff = 0;
            break;
        case ios::cur:
            newoff = (pos_in ? gptr() : pptr()) - eback();
            break;
        case ios::end:
            newoff = seekhigh - eback();
            break;
        }
        newoff += __off;
        if (0 <= newoff && newoff <= seekhigh - eback())
        {
            char* newpos = eback() + newoff;
            if (pos_in)
                setg(eback(), newpos, _VSTD::max(newpos, egptr()));
            if (pos_out)
            {
                // min(pbase, newpos), newpos, epptr()
                __off = epptr() - newpos;
                setp(min(pbase(), newpos), epptr());
                pbump(static_cast<int>((epptr() - pbase()) - __off));
            }
            __p = newoff;
        }
    }
    return pos_type(__p);
}

strstreambuf::pos_type
strstreambuf::seekpos(pos_type __sp, ios_base::openmode __which)
{
    off_type __p(-1);
    bool pos_in = __which & ios::in;
    bool pos_out = __which & ios::out;
    if (pos_in || pos_out)
    {
        if (!((pos_in && gptr() == nullptr) || (pos_out && pptr() == nullptr)))
        {
            off_type newoff = __sp;
            char* seekhigh = epptr() ? epptr() : egptr();
            if (0 <= newoff && newoff <= seekhigh - eback())
            {
                char* newpos = eback() + newoff;
                if (pos_in)
                    setg(eback(), newpos, _VSTD::max(newpos, egptr()));
                if (pos_out)
                {
                    // min(pbase, newpos), newpos, epptr()
                    off_type temp = epptr() - newpos;
                    setp(min(pbase(), newpos), epptr());
                    pbump(static_cast<int>((epptr() - pbase()) - temp));
                }
                __p = newoff;
            }
        }
    }
    return pos_type(__p);
}

istrstream::~istrstream()
{
}

ostrstream::~ostrstream()
{
}

strstream::~strstream()
{
}

_LIBCPP_END_NAMESPACE_STD
