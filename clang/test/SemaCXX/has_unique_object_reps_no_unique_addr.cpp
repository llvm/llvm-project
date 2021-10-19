// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -std=c++2a %s
//  expected-no-diagnostics

struct Empty {};

struct A {
  [[no_unique_address]] Empty e;
  char x;
};

static_assert(__has_unique_object_representations(A));

struct B {
  char x;
  [[no_unique_address]] Empty e;
};

static_assert(__has_unique_object_representations(B));

struct C {
  char x;
  [[no_unique_address]] Empty e1;
  [[no_unique_address]] Empty e2;
};

static_assert(!__has_unique_object_representations(C));

namespace TailPaddingReuse {
struct A {
private:
  int a;

public:
  char b;
};

struct B {
  [[no_unique_address]] A a;
  char c[3];
};
} // namespace TailPaddingReuse
static_assert(__has_unique_object_representations(TailPaddingReuse::B));
