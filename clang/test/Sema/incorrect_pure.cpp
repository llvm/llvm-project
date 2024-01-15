// RUN: %clang_cc1 -fsyntax-only -verify %s

[[gnu::pure]] void foo(); // expected-warning{{'pure' attribute on function returning 'void'}}

struct A {
    [[gnu::pure]] A(); // expected-warning{{constructor cannot be 'pure' (undefined behavior)}}
};
