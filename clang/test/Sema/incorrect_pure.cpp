// RUN: %clang_cc1 -fsyntax-only -verify %s

[[gnu::pure]] void foo(); // expected-warning{{'pure' attribute on function returning 'void'; attribute ignored}}

[[gnu::const]] void bar(); // expected-warning{{'const' attribute on function returning 'void'; attribute ignored}}

struct A {
    [[gnu::pure]] A(); // expected-warning{{'pure' attribute on function returning 'void'; attribute ignored}}

    [[gnu::const]] A(int); // expected-warning{{'const' attribute on function returning 'void'; attribute ignored}}
    [[gnu::pure]] ~A(); // expected-warning{{'pure' attribute on function returning 'void'; attribute ignored}}

    [[gnu::const]] [[gnu::pure]] int m(); // expected-warning{{'const' attribute imposes more restrictions; 'pure' attribute ignored}}
};
