// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -pedantic -verify -fsyntax-only
// expected-no-diagnostics

// Extract from PR38614
struct C {};

C f1() {
 return C{};
}

C f2(){
    C c;
    return c;
}
