// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s

// expected-no-diagnostics

// This test should not crash.
int f1( unsigned ) { return 0; }

template <class R, class... Args>
struct S1 {
    S1( R(*f)(Args...) ) {}
};

int main() {
    S1 s1( f1 );
}
