//===------------------------ stdexcept.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "stdexcept"
#include "new"
#include "string"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include "system_error"

// Note:  optimize for size

#pragma GCC visibility push(hidden)

namespace
{

class __libcpp_nmstr
{
private:
    const char* str_;

    typedef std::size_t unused_t;
    typedef std::int32_t count_t;

    static const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(2*sizeof(unused_t) +
                                                                       sizeof(count_t));

    count_t& count() const _NOEXCEPT {return (count_t&)(*(str_ - sizeof(count_t)));}
public:
    explicit __libcpp_nmstr(const char* msg);
    __libcpp_nmstr(const __libcpp_nmstr& s) _LIBCPP_CANTTHROW;
    __libcpp_nmstr& operator=(const __libcpp_nmstr& s) _LIBCPP_CANTTHROW;
    ~__libcpp_nmstr() _LIBCPP_CANTTHROW;
    const char* c_str() const _NOEXCEPT {return str_;}
};

__libcpp_nmstr::__libcpp_nmstr(const char* msg)
{
    std::size_t len = strlen(msg);
    str_ = new char[len + 1 + offset];
    unused_t* c = (unused_t*)str_;
    c[0] = c[1] = len;
    str_ += offset;
    count() = 0;
    std::strcpy(const_cast<char*>(c_str()), msg);
}

inline
__libcpp_nmstr::__libcpp_nmstr(const __libcpp_nmstr& s)
    : str_(s.str_)
{
    __sync_add_and_fetch(&count(), 1);
}

__libcpp_nmstr&
__libcpp_nmstr::operator=(const __libcpp_nmstr& s)
{
    const char* p = str_;
    str_ = s.str_;
    __sync_add_and_fetch(&count(), 1);
    if (__sync_add_and_fetch((count_t*)(p-sizeof(count_t)), -1) < 0)
        delete [] (p-offset);
    return *this;
}

inline
__libcpp_nmstr::~__libcpp_nmstr()
{
    if (__sync_add_and_fetch(&count(), -1) < 0)
        delete [] (str_ - offset);
}

}

#pragma GCC visibility pop

namespace std  // purposefully not using versioning namespace
{

logic_error::logic_error(const string& msg)
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr(msg.c_str());
}

logic_error::logic_error(const char* msg)
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr(msg);
}

logic_error::logic_error(const logic_error& le) _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr((const __libcpp_nmstr&)le.__imp_);
}

logic_error&
logic_error::operator=(const logic_error& le) _NOEXCEPT
{
    __libcpp_nmstr& s1 = (__libcpp_nmstr&)__imp_;
    const __libcpp_nmstr& s2 = (const __libcpp_nmstr&)le.__imp_;
    s1 = s2;
    return *this;
}

logic_error::~logic_error() _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    s.~__libcpp_nmstr();
}

const char*
logic_error::what() const _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    return s.c_str();
}

runtime_error::runtime_error(const string& msg)
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr(msg.c_str());
}

runtime_error::runtime_error(const char* msg)
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr(msg);
}

runtime_error::runtime_error(const runtime_error& le) _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    ::new(&s) __libcpp_nmstr((const __libcpp_nmstr&)le.__imp_);
}

runtime_error&
runtime_error::operator=(const runtime_error& le) _NOEXCEPT
{
    __libcpp_nmstr& s1 = (__libcpp_nmstr&)__imp_;
    const __libcpp_nmstr& s2 = (const __libcpp_nmstr&)le.__imp_;
    s1 = s2;
    return *this;
}

runtime_error::~runtime_error() _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    s.~__libcpp_nmstr();
}

const char*
runtime_error::what() const _NOEXCEPT
{
    __libcpp_nmstr& s = (__libcpp_nmstr&)__imp_;
    return s.c_str();
}

domain_error::~domain_error() _NOEXCEPT {}
invalid_argument::~invalid_argument() _NOEXCEPT {}
length_error::~length_error() _NOEXCEPT {}
out_of_range::~out_of_range() _NOEXCEPT {}

range_error::~range_error() _NOEXCEPT {}
overflow_error::~overflow_error() _NOEXCEPT {}
underflow_error::~underflow_error() _NOEXCEPT {}

}  // std
