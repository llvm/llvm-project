// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// If the initializer is (), the object is value-initialized.

// expected-no-diagnostics
namespace GH69890 {
struct A {
    constexpr A() {}
    int x;
};

struct B : A {
    int y;
};

static_assert(B().x == 0);
static_assert(B().y == 0);
}
