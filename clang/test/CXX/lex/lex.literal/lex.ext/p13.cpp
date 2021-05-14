// RUN: %clang_cc1 -std=c++2a -verify %s

template <typename T, T... cs> struct check; // expected-note {{template is declared here}}
template <>
struct check<char8_t, 34, 209, 130, 208, 181, 209, 129, 209, 130, 32, 240, 144, 128, 128> {};
template <typename T, T... str> int operator""_x() { // #1 expected-warning {{string literal operator templates are a GNU extension}}
  check<T, str...> chars;                            // expected-error {{implicit instantiation of undefined template 'check<char8_t, u8'"', u8'\xd1', u8'\x82', u8'\xd0', u8'\xb5', u8'\xd1', u8'\x81', u8'\xd1', u8'\x82', u8'_', u8'\xf0', u8'\x90', u8'\x80', u8'\x80'>'}}
  return 1;
}
int a = u8"\"тест 𐀀"_x;
int b = u8"\"тест_𐀀"_x; // expected-note {{in instantiation of function template specialization 'operator""_x<char8_t, u8'"', u8'\xd1', u8'\x82', u8'\xd0', u8'\xb5', u8'\xd1', u8'\x81', u8'\xd1', u8'\x82', u8'_', u8'\xf0', u8'\x90', u8'\x80', u8'\x80'>' requested here}}

template <auto> struct C{};
C<u8'x'>::D d; // expected-error {{no type named 'D' in 'C<u8'x'>'}}
