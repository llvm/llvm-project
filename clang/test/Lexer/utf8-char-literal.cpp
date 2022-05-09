// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c11 -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c2x -x c -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++1z -fsyntax-only -verify %s

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
#elif __STDC_VERSION__ > 202000L
char a = u8'ñ';      // expected-error {{character too large for enclosing character literal type}}
char b = u8'\x80';   // ok
char c = u8'\u0080'; // expected-error {{universal character name refers to a control character}}
char d = u8'\u1234'; // expected-error {{character too large for enclosing character literal type}}
char e = u8'ሴ';      // expected-error {{character too large for enclosing character literal type}}
char f = u8'ab';     // expected-error {{Unicode character literals may not contain multiple characters}}
_Static_assert(
    _Generic(u8'a',
             default : 0,
             unsigned char : 1),
    "Surprise!");
#endif
