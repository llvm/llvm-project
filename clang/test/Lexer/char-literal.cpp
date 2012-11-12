// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -Wfour-char-constants -fsyntax-only -verify %s

int a = 'ab'; // expected-warning {{multi-character character constant}}
int b = '\xFF\xFF'; // expected-warning {{multi-character character constant}}
int c = 'APPS'; // expected-warning {{multi-character character constant}}

char d = '⌘'; // expected-error {{character too large for enclosing character literal type}}
char e = '\u2318'; // expected-error {{character too large for enclosing character literal type}}

auto f = '\xE2\x8C\x98'; // expected-warning {{multi-character character constant}}

char16_t g = u'ab'; // expected-error {{Unicode character literals may not contain multiple characters}}
char16_t h = u'\U0010FFFD'; // expected-error {{character too large for enclosing character literal type}}

wchar_t i = L'ab'; // expected-warning {{extraneous characters in character constant ignored}}
wchar_t j = L'\U0010FFFD';

char32_t k = U'\U0010FFFD';

char l = 'Ø'; // expected-error {{character too large for enclosing character literal type}}
char m = '👿'; // expected-error {{character too large for enclosing character literal type}}

char32_t n = U'ab'; // expected-error {{Unicode character literals may not contain multiple characters}}
char16_t o = '👽'; // expected-error {{character too large for enclosing character literal type}}

char16_t p[2] = u"\U0000FFFF";
char16_t q[2] = u"\U00010000"; // expected-error {{too long}}
