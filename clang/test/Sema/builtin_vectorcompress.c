// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -disable-llvm-passes %s

typedef int int4 __attribute__((vector_size(16)));
typedef float float8 __attribute__((vector_size(32)));
typedef _Bool bitvec4 __attribute__((ext_vector_type(4)));
typedef _Bool bitvec8 __attribute__((ext_vector_type(8)));

void test_builtin_vectorelements(int4 vec1, float8 vec2, bitvec4 mask1, bitvec8 mask2, int4 passthru1, float8 passthru2) {
  // wrong number of arguments
  __builtin_experimental_vectorcompress(vec1); // expected-error {{too few arguments to function call}}
  __builtin_experimental_vectorcompress(vec1, mask2, passthru1, passthru1); // expected-error {{too many arguments to function call}}

  // valid
  (void) __builtin_experimental_vectorcompress(vec1, mask1);
  (void) __builtin_experimental_vectorcompress(vec1, mask1, passthru1);
  (void) __builtin_experimental_vectorcompress(vec2, mask2);
  (void) __builtin_experimental_vectorcompress(vec2, mask2, passthru2);

  // type mismatch
  __builtin_experimental_vectorcompress(vec1, mask2); // expected-error {{vector operands do not have the same number of elements}}
  __builtin_experimental_vectorcompress(vec2, mask1); // expected-error {{vector operands do not have the same number of elements}}
  __builtin_experimental_vectorcompress(vec1, mask1, passthru2); // expected-error {{arguments are of different types}}

  // invalid types
  int a;
  __builtin_experimental_vectorcompress(a, mask1, passthru1); // expected-error {{1st argument must be a vector type (was 'int')}}
  __builtin_experimental_vectorcompress(vec1, a, passthru1); // expected-error {{2nd argument must be a vector type (was 'int')}}
  __builtin_experimental_vectorcompress(vec1, mask1, a); // expected-error {{arguments are of different types}}
}

