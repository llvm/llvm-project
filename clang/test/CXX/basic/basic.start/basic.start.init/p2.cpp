// RUN: %clang_cc1 -verify %s -pedantic-errors -std=c++11
// RUN: %clang_cc1 -verify %s -pedantic-errors -std=c++14
// expected-no-diagnostics

struct foo_t {
  union {
    int i;
    volatile int j;
  } u;
};

__attribute__((__require_constant_initialization__))
static const foo_t x = {{0}};

union foo_u {
  int i;
  volatile int j;
};

__attribute__((__require_constant_initialization__))
static const foo_u y = {0};
