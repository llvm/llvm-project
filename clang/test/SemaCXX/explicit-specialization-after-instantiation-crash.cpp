// Test that explicit template specialization after instantiation
// is handled gracefully without assertion failure.
//
// RUN: not --crash %clang_cc1 -fsyntax-only -verify %s 2>&1 | FileCheck %s
//
// REQUIRES: asserts

// This test case triggers an assertion failure in getASTRecordLayout()
// when the compiler encounters explicit specialization after instantiation
// and attempts to get layout information during error recovery.

template <typename T>
struct X {
    struct Y {
        Y() : v(0) {}
        int v;
        int getValue();
    } y;
};

template <typename T>
int X<T>::Y::getValue() {
    return ++v;
}

// expected-error@+1 {{explicit specialization of 'Y' after instantiation}}
template <> struct X<int>::Y { int getValue() { return 55; } };
// expected-note@-11 {{implicit instantiation first required here}}

extern template class X<int>::Y;

int main() {
    X<int> x;
    return x.y.getValue(); // expected-error {{no member named 'getValue'}}
}

// Verify the assertion failure occurs
// CHECK: Assertion
// CHECK: isInvalidDecl
// CHECK: Cannot get layout of invalid decl