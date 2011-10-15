//===------------------------- string.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "string"
#include "cstdlib"
#include "cwchar"
#include "cerrno"
#if _WIN32
#include "support/win32/support.h"
#endif // _WIN32

_LIBCPP_BEGIN_NAMESPACE_STD

template class __basic_string_common<true>;

template class basic_string<char>;
template class basic_string<wchar_t>;

template
    string
    operator+<char, char_traits<char>, allocator<char> >(char const*, string const&);

int
stoi(const string& str, size_t* idx, int base)
{
    char* ptr;
    const char* const p = str.c_str();
    long r = strtol(p, &ptr, base);
    if (r < numeric_limits<int>::min() || numeric_limits<int>::max() < r)
        ptr = const_cast<char*>(p);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoi: no conversion");
        throw out_of_range("stoi: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return static_cast<int>(r);
}

int
stoi(const wstring& str, size_t* idx, int base)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    long r = wcstol(p, &ptr, base);
    if (r < numeric_limits<int>::min() || numeric_limits<int>::max() < r)
        ptr = const_cast<wchar_t*>(p);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoi: no conversion");
        throw out_of_range("stoi: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return static_cast<int>(r);
}

long
stol(const string& str, size_t* idx, int base)
{
    char* ptr;
    const char* const p = str.c_str();
    long r = strtol(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stol: no conversion");
        throw out_of_range("stol: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

long
stol(const wstring& str, size_t* idx, int base)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    long r = wcstol(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stol: no conversion");
        throw out_of_range("stol: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

unsigned long
stoul(const string& str, size_t* idx, int base)
{
    char* ptr;
    const char* const p = str.c_str();
    unsigned long r = strtoul(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoul: no conversion");
        throw out_of_range("stoul: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

unsigned long
stoul(const wstring& str, size_t* idx, int base)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    unsigned long r = wcstoul(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoul: no conversion");
        throw out_of_range("stoul: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

long long
stoll(const string& str, size_t* idx, int base)
{
    char* ptr;
    const char* const p = str.c_str();
    long long r = strtoll(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoll: no conversion");
        throw out_of_range("stoll: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

long long
stoll(const wstring& str, size_t* idx, int base)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    long long r = wcstoll(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoll: no conversion");
        throw out_of_range("stoll: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

unsigned long long
stoull(const string& str, size_t* idx, int base)
{
    char* ptr;
    const char* const p = str.c_str();
    unsigned long long r = strtoull(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoull: no conversion");
        throw out_of_range("stoull: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

unsigned long long
stoull(const wstring& str, size_t* idx, int base)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    unsigned long long r = wcstoull(p, &ptr, base);
    if (ptr == p)
    {
#ifndef _LIBCPP_NO_EXCEPTIONS
        if (r == 0)
            throw invalid_argument("stoull: no conversion");
        throw out_of_range("stoull: out of range");
#endif  // _LIBCPP_NO_EXCEPTIONS
    }
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

float
stof(const string& str, size_t* idx)
{
    char* ptr;
    const char* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    double r = strtod(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stof: out of range");
    if (ptr == p)
        throw invalid_argument("stof: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return static_cast<float>(r);
}

float
stof(const wstring& str, size_t* idx)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    double r = wcstod(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stof: out of range");
    if (ptr == p)
        throw invalid_argument("stof: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return static_cast<float>(r);
}

double
stod(const string& str, size_t* idx)
{
    char* ptr;
    const char* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    double r = strtod(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stod: out of range");
    if (ptr == p)
        throw invalid_argument("stod: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

double
stod(const wstring& str, size_t* idx)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    double r = wcstod(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stod: out of range");
    if (ptr == p)
        throw invalid_argument("stod: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

long double
stold(const string& str, size_t* idx)
{
    char* ptr;
    const char* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    long double r = strtold(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stold: out of range");
    if (ptr == p)
        throw invalid_argument("stold: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

long double
stold(const wstring& str, size_t* idx)
{
    wchar_t* ptr;
    const wchar_t* const p = str.c_str();
    int errno_save = errno;
    errno = 0;
    long double r = wcstold(p, &ptr);
    swap(errno, errno_save);
#ifndef _LIBCPP_NO_EXCEPTIONS
    if (errno_save == ERANGE)
        throw out_of_range("stold: out of range");
    if (ptr == p)
        throw invalid_argument("stold: no conversion");
#endif  // _LIBCPP_NO_EXCEPTIONS
    if (idx)
        *idx = static_cast<size_t>(ptr - p);
    return r;
}

string to_string(int val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%d", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(unsigned val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%u", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(long val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%ld", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(unsigned long val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%lu", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(long long val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%lld", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(unsigned long long val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%llu", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(float val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%f", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(double val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%f", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

string to_string(long double val)
{
    string s;
    s.resize(s.capacity());
    while (true)
    {
        int n2 = snprintf(&s[0], s.size()+1, "%Lf", val);
        if (n2 <= s.size())
        {
            s.resize(n2);
            break;
        }
        s.resize(n2);
    }
    return s;
}

wstring to_wstring(int val)
{
    const size_t n = (numeric_limits<int>::digits / 3)
          + ((numeric_limits<int>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%d", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(unsigned val)
{
    const size_t n = (numeric_limits<unsigned>::digits / 3)
          + ((numeric_limits<unsigned>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%u", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(long val)
{
    const size_t n = (numeric_limits<long>::digits / 3)
          + ((numeric_limits<long>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%ld", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(unsigned long val)
{
    const size_t n = (numeric_limits<unsigned long>::digits / 3)
          + ((numeric_limits<unsigned long>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%lu", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(long long val)
{
    const size_t n = (numeric_limits<long long>::digits / 3)
          + ((numeric_limits<long long>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%lld", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(unsigned long long val)
{
    const size_t n = (numeric_limits<unsigned long long>::digits / 3)
          + ((numeric_limits<unsigned long long>::digits % 3) != 0)
          + 1;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%llu", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(float val)
{
    const size_t n = 20;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%f", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(double val)
{
    const size_t n = 20;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%f", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

wstring to_wstring(long double val)
{
    const size_t n = 20;
    wstring s(n, wchar_t());
    s.resize(s.capacity());
    while (true)
    {
        int n2 = swprintf(&s[0], s.size()+1, L"%Lf", val);
        if (n2 > 0)
        {
            s.resize(n2);
            break;
        }
        s.resize(2*s.size());
        s.resize(s.capacity());
    }
    return s;
}

_LIBCPP_END_NAMESPACE_STD
