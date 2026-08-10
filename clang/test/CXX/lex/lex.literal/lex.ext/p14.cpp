// RUN: %clang_cc1 -std=c++2a -verify -triple x86_64-unknown-linux %s

template <typename T, T... cs> struct check; // expected-note {{template is declared here}}
template <>
struct check<wchar_t, 34, 1090, 1077, 1089, 1090, 32, 65536> {};
template <typename T, T... str> int operator""_x() { // #1 expected-warning {{string literal operator templates are a GNU extension}}
  check<T, str...> chars;                            // expected-error {{implicit instantiation of undefined template 'check<wchar_t, L'"', L'\u0442', L'\u0435', L'\u0441', L'\u0442', L'_', L'\U00010000'>'}}
  return 1;
}
void *operator""_x(const char *); // #2
int h = LR"("тест_𐀀)"_x;          // expected-note {{in instantiation of function template specialization 'operator""_x<wchar_t, L'"', L'\u0442', L'\u0435', L'\u0441', L'\u0442', L'_', L'\U00010000'>' requested here}}
