// RUN: %clang_cc1 -fsyntax-only %s
typedef struct foo foo;

void blah(int foo) {
  foo = 1;
}
