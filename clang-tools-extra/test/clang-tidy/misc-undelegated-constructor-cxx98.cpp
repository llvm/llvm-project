// RUN: clang-tidy %s -checks=-*,misc-undelegated-constructor -- -std=c++98 | count 0

// Note: this test expects no diagnostics, but FileCheck cannot handle that,
// hence the use of | count 0.

struct Ctor;
Ctor foo();

struct Ctor {
  Ctor();
  Ctor(int);
  Ctor(int, int);
  Ctor(Ctor *i) {
    Ctor();
    Ctor(0);
    Ctor(1, 2);
    foo();
  }
};

Ctor::Ctor() {
  Ctor(1);
}
