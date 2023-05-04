// RUN: %clang_cc1 -fsyntax-only -verify=expected -std=c++23 -Wall %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,expected-cxx20 -std=c++20 -Wall %s

namespace ns {
    int i;
    enum class e {};
}
void f() {

    for (using foo = int;true;); //expected-cxx20-warning {{alias declaration in this context is a C++23 extension}}

    switch(using foo = int; 0) { //expected-cxx20-warning {{alias declaration in this context is a C++23 extension}}
        case 0: break;
    }

    if(using foo = int; false) {} //expected-cxx20-warning {{alias declaration in this context is a C++23 extension}}


    if (using enum ns::e; false){}  // expected-error {{expected '='}}

    for (using ns::i; true;);  // expected-error {{expected '='}}

    if (using ns::i; false){}  // expected-error {{expected '='}}

    switch(using ns::i; 0) {   // expected-error {{expected '='}}
        case 0: break;
    }

}
