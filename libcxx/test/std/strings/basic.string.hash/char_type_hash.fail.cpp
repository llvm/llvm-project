//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// UNSUPPORTED: c++03

// <string>

// Test that hash specializations for <string> require "char_traits<_CharT>" not just any "_Trait".

#include <functional>
#include <string>

#include "test_macros.h"

template <class _CharT>
struct trait // copied from <__string>
{
    typedef _CharT         char_type;
    typedef int            int_type;
    typedef std::streamoff off_type;
    typedef std::streampos pos_type;
    typedef std::mbstate_t state_type;

    static inline void assign(char_type& __c1, const char_type& __c2) {
        __c1 = __c2;
    }
    static inline bool eq(char_type __c1, char_type __c2) { return __c1 == __c2; }
    static inline bool lt(char_type __c1, char_type __c2) { return __c1 < __c2; }

    static int compare(const char_type* __s1, const char_type* __s2, size_t __n);
    static size_t length(const char_type* __s);
    static const char_type* find(const char_type* __s, size_t __n,
                                 const char_type& __a);

    static char_type* move(char_type* __s1, const char_type* __s2, size_t __n);
    static char_type* copy(char_type* __s1, const char_type* __s2, size_t __n);
    static char_type* assign(char_type* __s, size_t __n, char_type __a);

    static inline int_type not_eof(int_type __c) {
        return eq_int_type(__c, eof()) ? ~eof() : __c;
    }
    static inline char_type to_char_type(int_type __c) { return char_type(__c); }
    static inline int_type to_int_type(char_type __c) { return int_type(__c); }
    static inline bool eq_int_type(int_type __c1, int_type __c2) {
        return __c1 == __c2;
    }
    static inline int_type eof() { return int_type(EOF); }
};

template <class CharT>
void test() {
    typedef std::basic_string<CharT, trait<CharT> > str_t;
    std::hash<str_t>
        h; // expected-error-re 4 {{{{call to implicitly-deleted default constructor of 'std::hash<str_t>'|implicit instantiation of undefined template}} {{.+}}}}}}
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    // expected-error-re@-2 {{{{call to implicitly-deleted default constructor of 'std::hash<str_t>'|implicit instantiation of undefined template}} {{.+}}}}}}
#endif
    (void)h;
}

int main(int, char**) {
    test<char>();
    test<wchar_t>();
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
    test<char8_t>();
#endif
    test<char16_t>();
    test<char32_t>();

    return 0;
}
