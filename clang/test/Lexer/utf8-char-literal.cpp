// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c11 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c2x -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++17 -fsyntax-only -fchar8_t -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++20 -fsyntax-only -fno-char8_t -verify %s

int array0[u'ñ' == u'\xf1'? 1 : -1];
int array1['\xF1' !=  u'\xf1'? 1 : -1];
int array1['ñ' !=  u'\xf1'? 1 : -1]; // expected-error {{character too large for enclosing character literal type}}
#if __cplusplus > 201402L
char a = u8'ñ'; // expected-error {{character too large for enclosing character literal type}}
char b = u8'\x80'; // ok
char c = u8'\u0080'; // expected-error {{character too large for enclosing character literal type}}
char d = u8'\u1234'; // expected-error {{character too large for enclosing character literal type}}
char e = u8'ሴ'; // expected-error {{character too large for enclosing character literal type}}
char f = u8'ab'; // expected-error {{Unicode character literals may not contain multiple characters}}
#elif __STDC_VERSION__ >= 202311L
char a = u8'ñ';      // expected-error {{character too large for enclosing character literal type}}
char b = u8'\x80';   // ok
char c = u8'\u0000'; // ok
char d = u8'\u1234'; // expected-error {{character too large for enclosing character literal type}}
char e = u8'ሴ';      // expected-error {{character too large for enclosing character literal type}}
char f = u8'ab';     // expected-error {{Unicode character literals may not contain multiple characters}}
_Static_assert(
    _Generic(u8'a',
             default : 0,
             unsigned char : 1),
    "Surprise!");
#endif


// UTF-8 character literals are enabled in C++17 and later. If `-fchar8_t` is not enabled
// (as is the case in C++17), then UTF-8 character literals may produce signed or
// unsigned values depending on whether char is a signed type. If `-fchar8_t` is enabled
// (which is the default behavior for C++20), then UTF-8 character literals always
// produce unsigned values. The tests below depend on the target having a signed
// 8-bit char so that '\xff' produces a negative value.
#if __cplusplus >= 201703L
#  if !defined(__cpp_char8_t)
#    if !(u8'\xff' == '\xff')
#      error UTF-8 character value did not match ordinary character literal; this is unexpected
#    endif
#  else
#    if u8'\xff' == '\xff' // expected-warning {{right side of operator converted from negative value to unsigned}}
#      error UTF-8 character value matched ordinary character literal; this is unexpected
#    endif
#  endif
#endif

/// In C23, u8 char literals are always unsigned.
#if __STDC_VERSION__ >= 202311L
#  if u8'\xff' == '\xff'// expected-warning {{right side of operator converted from negative value to unsigned}}
#    error u8 char literal is not unsigned
#  endif
#endif
