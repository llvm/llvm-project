// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T> requires (sizeof(T) >= 4 && sizeof(T) <= 10)
// expected-note@-1{{because 'sizeof(char [20]) <= 10' (20 <= 10) evaluated to false}}
// expected-note@-2{{because 'sizeof(char) >= 4' (1 >= 4) evaluated to false}}
void foo() requires (sizeof(T) <= 8) {}
// expected-note@-1{{candidate template ignored: constraints not satisfied [with T = char]}}
// expected-note@-2{{candidate template ignored: constraints not satisfied [with T = char [9]]}}
// expected-note@-3{{candidate template ignored: constraints not satisfied [with T = char [20]]}}
// expected-note@-4{{because 'sizeof(char [9]) <= 8' (9 <= 8) evaluated to false}}

void bar() {
  foo<char>(); // expected-error{{no matching function for call to 'foo'}}
  foo<int>();
  foo<unsigned long long int>();
  foo<char[9]>(); // expected-error{{no matching function for call to 'foo'}}
  foo<char[20]>(); // expected-error{{no matching function for call to 'foo'}}
}