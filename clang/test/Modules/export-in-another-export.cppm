// RUN: %clang_cc1 -std=c++20 %s -fsyntax-only -verify
export module M;
export { // expected-note {{export block begins here}}
    export int foo() { return 43; } // expected-error {{export declaration appears within another export declaration}}
}
