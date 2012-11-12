// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wc++98-compat -Werror %s
// RUN: %clang_cc1 -fsyntax-only -std=c++98 -Werror %s

// -Wc++98-compat-pedantic warns on C++11 features which we accept without a
// warning in C++98 mode.

#line 32767 // ok
#line 32768 // expected-warning {{#line number greater than 32767 is incompatible with C++98}}

#define VA_MACRO(x, ...) x // expected-warning {{variadic macros are incompatible with C++98}}
VA_MACRO(,x) // expected-warning {{empty macro arguments are incompatible with C++98}}

; // expected-warning {{extra ';' outside of a function is incompatible with C++98}}

enum Enum {
  Enum_value, // expected-warning {{commas at the end of enumerator lists are incompatible with C++98}}
};

template<typename T> struct InstantiationAfterSpecialization {};
template<> struct InstantiationAfterSpecialization<int> {}; // expected-note {{here}}
template struct InstantiationAfterSpecialization<int>; // expected-warning {{explicit instantiation of 'InstantiationAfterSpecialization<int>' that occurs after an explicit specialization is incompatible with C++98}}

void *dlsym();
void (*FnPtr)() = (void(*)())dlsym(); // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}
void *FnVoidPtr = (void*)&dlsym; // expected-warning {{cast between pointer-to-function and pointer-to-object is incompatible with C++98}}

struct ConvertToInt {
  operator int();
};
int *ArraySizeConversion = new int[ConvertToInt()]; // expected-warning {{implicit conversion from array size expression of type 'ConvertToInt' to integral type 'int' is incompatible with C++98}}

template<typename T> class ExternTemplate {};
extern template class ExternTemplate<int>; // expected-warning {{extern templates are incompatible with C++98}}

long long ll1 = // expected-warning {{'long long' is incompatible with C++98}}
         -42LL; // expected-warning {{'long long' is incompatible with C++98}}
unsigned long long ull1 = // expected-warning {{'long long' is incompatible with C++98}}
                   42ULL; // expected-warning {{'long long' is incompatible with C++98}}

