// RUN: %clang_cc1 -std=c++14 -verify %s

// Test for GH#199527 - enable_if attribute on member function referencing
// incomplete class type should not crash during class body parsing.

struct S {
    ~S() {}
    bool b;
    void foo(S b) __attribute__((enable_if(b.b, ""))); // expected-no-diagnostics
};