// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify %s

void test_builtin_vectorelements() {
  __builtin_vectorelements(int); // expected-error {{'__builtin_vectorelements' argument must be a vector}}
  __builtin_vectorelements(float); // expected-error {{'__builtin_vectorelements' argument must be a vector}}
  __builtin_vectorelements(long*); // expected-error {{'__builtin_vectorelements' argument must be a vector}}

  int a;
  __builtin_vectorelements(a); // expected-error {{'__builtin_vectorelements' argument must be a vector}}

  typedef int veci4 __attribute__((vector_size(16)));
  (void) __builtin_vectorelements(veci4);

  veci4 vec;
  (void) __builtin_vectorelements(vec);

  typedef veci4 some_other_vec;
  (void) __builtin_vectorelements(some_other_vec);

  struct Foo { int a; };
  __builtin_vectorelements(struct Foo); // expected-error {{'__builtin_vectorelements' argument must be a vector}}
}

