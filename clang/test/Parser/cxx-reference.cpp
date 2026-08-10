// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

extern char *bork;
char *& bar = bork;

int val;

void foo(int &a) {
}

typedef int & A;

void g(const A aref) { // expected-warning {{'const' qualifier on reference type 'A' (aka 'int &') has no effect}}
}

int & const X = val; // expected-error {{'const' qualifier may not be applied to a reference}}
int & volatile Y = val; // expected-error {{'volatile' qualifier may not be applied to a reference}}
int & const volatile Z = val; /* expected-error {{'const' qualifier may not be applied}} \
                           expected-error {{'volatile' qualifier may not be applied}} */

typedef int && RV; 
#if __cplusplus <= 199711L
// expected-warning@-2 {{rvalue references are a C++11 extension}}
#endif
