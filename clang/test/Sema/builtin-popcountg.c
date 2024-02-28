// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wpedantic %s

typedef int int2 __attribute__((ext_vector_type(2)));

void test_builtin_popcountg(int i, double d, int2 i2) {
  __builtin_popcountg();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_popcountg(i, i);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  __builtin_popcountg(d);
  // expected-error@-1 {{1st argument must be a type of integer (was 'double')}}
  __builtin_popcountg(i2);
  // expected-error@-1 {{1st argument must be a type of integer (was 'int2' (vector of 2 'int' values))}}
}
