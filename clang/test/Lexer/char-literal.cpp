// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++03 -Wfour-char-constants -fsyntax-only -verify=cxx03,expected %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -Wfour-char-constants -fsyntax-only -verify=cxx,expected %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++17 -Wfour-char-constants -fsyntax-only -verify=cxx,expected %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++20 -Wfour-char-constants -fsyntax-only -verify=cxx,expected %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c11 -x c -Wfour-char-constants -fsyntax-only -verify=c11,expected %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c2x -x c -Wfour-char-constants -fsyntax-only -verify=c2x,expected %s

#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
typedef __CHAR16_TYPE__ char16_t;
typedef __CHAR32_TYPE__ char32_t;
#endif

int a = 'ab'; // expected-warning {{multi-character character constant}}
int b = '\xFF\xFF'; // expected-warning {{multi-character character constant}}
int c = 'APPS'; // expected-warning {{multi-character character constant}}

char d = '⌘'; // expected-error {{character too large for enclosing character literal type}}
char e = '\u2318'; // expected-error {{character too large for enclosing character literal type}}

#if !defined(__cplusplus) || __cplusplus > 201100L
#ifdef __cplusplus
auto f = '\xE2\x8C\x98'; // expected-warning {{multi-character character constant}}
#endif

char16_t g = u'ab'; // expected-error {{Unicode character literals may not contain multiple characters}}
char16_t h = u'\U0010FFFD'; // expected-error {{character too large for enclosing character literal type}}

wchar_t i = L'ab'; // expected-error {{wide character literals may not contain multiple characters}}

wchar_t j = L'\U0010FFFD';

char32_t k = U'\U0010FFFD';

char l = 'Ø'; // expected-error {{character too large for enclosing character literal type}}
char m = '👿'; // expected-error {{character too large for enclosing character literal type}}

char32_t n = U'ab'; // expected-error {{Unicode character literals may not contain multiple characters}}
char16_t o = '👽'; // expected-error {{character too large for enclosing character literal type}}

char16_t p[2] = u"\U0000FFFF";
char16_t q[2] = u"\U00010000";
#ifdef __cplusplus
// expected-error@-2 {{too long}}
#endif

// UTF-8 character literal code point ranges.
#if __cplusplus >= 201703L || __STDC_VERSION__ >= 201710L
_Static_assert(u8'\U00000000' == 0x00, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(u8'\U0000007F' == 0x7F, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(u8'\U00000080', ""); // c11-error {{universal character name refers to a control character}}
                                    // cxx-error@-1 {{character too large for enclosing character literal type}}
                                    // c2x-error@-2 {{character too large for enclosing character literal type}}
_Static_assert((unsigned char)u8'\xFF' == (unsigned char)0xFF, "");
#endif

// UTF-8 string literal code point ranges.
_Static_assert(u8"\U00000000"[0] == 0x00, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(u8"\U0000007F"[0] == 0x7F, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert((unsigned char)u8"\U00000080"[0] == (unsigned char)0xC2, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert((unsigned char)u8"\U00000080"[1] == (unsigned char)0x80, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert((unsigned char)u8"\U000007FF"[0] == (unsigned char)0xDF, "");
_Static_assert((unsigned char)u8"\U000007FF"[1] == (unsigned char)0xBF, "");
_Static_assert((unsigned char)u8"\U00000800"[0] == (unsigned char)0xE0, "");
_Static_assert((unsigned char)u8"\U00000800"[1] == (unsigned char)0xA0, "");
_Static_assert((unsigned char)u8"\U00000800"[2] == (unsigned char)0x80, "");
_Static_assert(u8"\U0000D800"[0], ""); // expected-error {{invalid universal character}}
_Static_assert(u8"\U0000DFFF"[0], ""); // expected-error {{invalid universal character}}
_Static_assert((unsigned char)u8"\U0000FFFF"[0] == (unsigned char)0xEF, "");
_Static_assert((unsigned char)u8"\U0000FFFF"[1] == (unsigned char)0xBF, "");
_Static_assert((unsigned char)u8"\U0000FFFF"[2] == (unsigned char)0xBF, "");
_Static_assert((unsigned char)u8"\U00010000"[0] == (unsigned char)0xF0, "");
_Static_assert((unsigned char)u8"\U00010000"[1] == (unsigned char)0x90, "");
_Static_assert((unsigned char)u8"\U00010000"[2] == (unsigned char)0x80, "");
_Static_assert((unsigned char)u8"\U00010000"[3] == (unsigned char)0x80, "");
_Static_assert((unsigned char)u8"\U0010FFFF"[0] == (unsigned char)0xF4, "");
_Static_assert((unsigned char)u8"\U0010FFFF"[1] == (unsigned char)0x8F, "");
_Static_assert((unsigned char)u8"\U0010FFFF"[2] == (unsigned char)0xBF, "");
_Static_assert((unsigned char)u8"\U0010FFFF"[3] == (unsigned char)0xBF, "");
_Static_assert(u8"\U00110000"[0], ""); // expected-error {{invalid universal character}}

#if !defined(__STDC_UTF_16__)
#error __STDC_UTF_16__ is not defined.
#endif
#if __STDC_UTF_16__ != 1
#error __STDC_UTF_16__ has the wrong value.
#endif

// UTF-16 character literal code point ranges.
_Static_assert(u'\U00000000' == 0x0000, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(u'\U0000D800', ""); // expected-error {{invalid universal character}}
_Static_assert(u'\U0000DFFF', ""); // expected-error {{invalid universal character}}
_Static_assert(u'\U0000FFFF' == 0xFFFF, "");
_Static_assert(u'\U00010000', ""); // expected-error {{character too large for enclosing character literal type}}

// UTF-16 string literal code point ranges.
_Static_assert(u"\U00000000"[0] == 0x0000, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(u"\U0000D800"[0], ""); // expected-error {{invalid universal character}}
_Static_assert(u"\U0000DFFF"[0], ""); // expected-error {{invalid universal character}}
_Static_assert(u"\U0000FFFF"[0] == 0xFFFF, "");
_Static_assert(u"\U00010000"[0] == 0xD800, "");
_Static_assert(u"\U00010000"[1] == 0xDC00, "");
_Static_assert(u"\U0010FFFF"[0] == 0xDBFF, "");
_Static_assert(u"\U0010FFFF"[1] == 0xDFFF, "");
_Static_assert(u"\U00110000"[0], ""); // expected-error {{invalid universal character}}

#if !defined(__STDC_UTF_32__)
#error __STDC_UTF_32__ is not defined.
#endif
#if __STDC_UTF_32__ != 1
#error __STDC_UTF_32__ has the wrong value.
#endif

// UTF-32 character literal code point ranges.
_Static_assert(U'\U00000000' == 0x00000000, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(U'\U0010FFFF' == 0x0010FFFF, "");
_Static_assert(U'\U00110000', ""); // expected-error {{invalid universal character}}

// UTF-32 string literal code point ranges.
_Static_assert(U"\U00000000"[0] == 0x00000000, ""); // c11-error {{universal character name refers to a control character}}
_Static_assert(U"\U0000D800"[0], ""); // expected-error {{invalid universal character}}
_Static_assert(U"\U0000DFFF"[0], ""); // expected-error {{invalid universal character}}
_Static_assert(U"\U0010FFFF"[0] == 0x0010FFFF, "");
_Static_assert(U"\U00110000"[0], ""); // expected-error {{invalid universal character}}

#endif // !defined(__cplusplus) || __cplusplus > 201100L

_Static_assert('\u0024' == '$', "");
_Static_assert('\u0040' == '@', "");
_Static_assert('\u0060' == '`', "");

_Static_assert('\u0061' == 'a', "");  // c11-error {{character 'a' cannot be specified by a universal character name}} \
                                      // cxx03-error {{character 'a' cannot be specified by a universal character name}}
_Static_assert('\u0000' == '\0', ""); // c11-error {{universal character name refers to a control character}} \
                                      // cxx03-error {{universal character name refers to a control character}}
