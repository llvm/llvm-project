// RUN: %clang_cc1 -fsyntax-only -verify %s

// GH106358: Test that preferred_name propagates to specializations 
// instantiated before the attribute is seen.

template<typename T>
void foo(T arg) {} // expected-note {{candidate function template not viable: no known conversion from 'int' to 'my_string' (aka 'my_basic_string<char>') for 1st argument}}

template<typename T> struct my_basic_string;
using my_string = my_basic_string<char>;

// This attribute application now correctly propagates to the existing specialization
template<typename T> 
struct __attribute__((__preferred_name__(my_string))) my_basic_string {};

int main() {
    foo<my_string>(123); // expected-error {{no matching function for call to 'foo'}}
}
