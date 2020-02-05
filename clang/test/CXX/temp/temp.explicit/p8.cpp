// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

template<typename T, typename S = char> requires (sizeof(T) + sizeof(S) < 10)
// expected-note@-1{{because 'sizeof(char [100]) + sizeof(char) < 10' (101 < 10) evaluated to false}}
void f(T t, S s) requires (sizeof(t) == 1 && sizeof(s) == 1) { };
// expected-note@-1{{candidate template ignored: constraints not satisfied [with T = int, S = char]}}
// expected-note@-2{{because 'sizeof (t) == 1' (4 == 1) evaluated to false}}
// expected-note@-3{{candidate template ignored: constraints not satisfied [with T = char, S = short]}}
// expected-note@-4{{because 'sizeof (s) == 1' (2 == 1) evaluated to false}}
// expected-note@-5{{candidate template ignored: constraints not satisfied [with T = char [100], S = char]}}

template<>
void f<int>(int t, char s) { };
// expected-error@-1{{no function template matches function template specialization 'f'}}

template<>
void f<char, short>(char t, short s) { };
// expected-error@-1{{no function template matches function template specialization 'f'}}

template<>
void f<char[100]>(char t[100], char s) { };
// expected-error@-1{{no function template matches function template specialization 'f'}}