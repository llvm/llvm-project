// RUN: %clang_cc1 -std=c23 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wpedantic %s

typedef int int2 __attribute__((ext_vector_type(2)));

void test_builtin_popcountg(short s, int i, __int128 i128, _BitInt(128) bi128,
                            double d, int2 i2) {
  __builtin_popcountg();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_popcountg(i, i);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  __builtin_popcountg(s);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was 'short')}}
  __builtin_popcountg(i);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was 'int')}}
  __builtin_popcountg(i128);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was '__int128')}}
  __builtin_popcountg(bi128);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was '_BitInt(128)')}}
  __builtin_popcountg(d);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was 'double')}}
  __builtin_popcountg(i2);
  // expected-error@-1 {{1st argument must be a type of unsigned integer (was 'int2' (vector of 2 'int' values))}}
}
