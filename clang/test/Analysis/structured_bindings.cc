// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core -verify %s

// expected-no-diagnostics
struct s { int a; };
int foo() {
    auto[a] = s{1}; // FIXME: proper modelling
    if (a) {
    }
}

