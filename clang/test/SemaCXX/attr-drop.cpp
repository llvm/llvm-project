// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

template<typename T>
T drop([[clang::drop]] T& value);

void foo() {
  int x = 42;
  drop(x); // expected-note {{'x' dropped here}}
  int y = x; // expected-error{{use of 'x' after it was dropped}}
}

template<typename A, typename B>
void bar(A, B);

void foo2() {
    int x = 42;
    bar(x, drop(x));
    int y = 42;
    bar(drop(y), y); // expected-error {{use of 'y' after it was dropped}} expected-note {{'y' dropped here}}
}

struct type {
    int x = 42;
};

int& baz();

void drop_nondropable() {
    drop(baz());
    int x = 42;
    type a;
    drop(a.x);
    drop(a.x);
    drop(x);
}
