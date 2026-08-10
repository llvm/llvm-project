// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -disable-llvm-passes %s

void test_builtin_vectorelements() {
  __builtin_vectorelements(int); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
  __builtin_vectorelements(float); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
  __builtin_vectorelements(long*); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

  int a;
  __builtin_vectorelements(a); // expected-error {{argument to __builtin_vectorelements must be of vector type}}

  typedef int veci4 __attribute__((vector_size(16)));
  (void) __builtin_vectorelements(veci4);

  veci4 vec;
  (void) __builtin_vectorelements(vec);

  typedef veci4 some_other_vec;
  (void) __builtin_vectorelements(some_other_vec);

  struct Foo { int a; };
  __builtin_vectorelements(struct Foo); // expected-error {{argument to __builtin_vectorelements must be of vector type}}
}

