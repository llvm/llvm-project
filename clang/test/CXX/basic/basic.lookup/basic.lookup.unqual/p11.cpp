// RUN: %clang_cc1 -fsyntax-only -verify %s

static const int a = 10;

void f0(int a, 
        int b = a) { // expected-error {{default argument references parameter 'a'}}
}

template<int a, 
         int b = a>
class A {
};
