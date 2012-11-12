// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

typedef decltype(sizeof(int)) size_t;

// FIXME: These diagnostics should say 'size_t' instead of 'unsigned long'
int a = 123_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'unsigned long long' or 'const char *', and no matching literal operator template}}
int b = 4.2_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'long double' or 'const char *', and no matching literal operator template}}
int c = "foo"_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with arguments of types 'const char *' and 'unsigned}}
int d = L"foo"_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with arguments of types 'const wchar_t *' and 'unsigned}}
int e = u8"foo"_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with arguments of types 'const char *' and 'unsigned}}
int f = u"foo"_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with arguments of types 'const char16_t *' and 'unsigned}}
int g = U"foo"_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with arguments of types 'const char32_t *' and 'unsigned}}
int h = 'y'_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'char'}}
int i = L'y'_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'wchar_t'}}
int j = u'y'_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'char16_t'}}
int k = U'y'_x; // expected-error {{no matching literal operator for call to 'operator "" _x' with argument of type 'char32_t'}}
