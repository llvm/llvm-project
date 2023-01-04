// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20
// expected-no-diagnostics

namespace GH59624 {

struct Foo{
    int x{0};
};

struct Bar{
    const Foo y;
};

// Deleted move assignment shouldn't make type non-trivially copyable:

static_assert(__is_trivially_copyable(Bar), "");

}
