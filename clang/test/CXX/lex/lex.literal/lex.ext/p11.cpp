// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

using size_t = decltype(sizeof(int));

template<typename T, typename U> struct same_type;
template<typename T> struct same_type<T, T> {};
template<typename T> using X = T;
template<typename CharT, X<CharT>...>
int operator ""_x(); // expected-warning {{string literal operator templates are a GNU extension}}
template<char...>
double operator ""_x();

auto a="string"_x;
auto b=42_x;
same_type<decltype(a), int> test_a;
same_type<decltype(b), double> test_b;

char operator ""_x(const char *begin, size_t size);
auto c="string"_x;
auto d=L"string"_x;
same_type<decltype(c), char> test_c;
same_type<decltype(d), int> test_d;
