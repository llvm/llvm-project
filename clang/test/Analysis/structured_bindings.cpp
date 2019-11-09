// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core -verify %s

struct s { int a; };
int foo() {
    auto[a] = s{1}; // FIXME: proper modelling
    if (a) {
    }
} // expected-warning{{non-void function does not return a value}}

