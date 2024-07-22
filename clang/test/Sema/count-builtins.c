// RUN: %clang_cc1 -std=c23 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wpedantic %s

typedef int int2 __attribute__((ext_vector_type(2)));

void test_builtin_popcountg(short s, int i, __int128 i128, _BitInt(128) bi128,
                            double d, int2 i2) {
  __builtin_popcountg();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_popcountg(i, i);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  __builtin_popcountg(s);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'short')}}
  __builtin_popcountg(i);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int')}}
  __builtin_popcountg(i128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '__int128')}}
  __builtin_popcountg(bi128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '_BitInt(128)')}}
  __builtin_popcountg(d);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'double')}}
  __builtin_popcountg(i2);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int2' (vector of 2 'int' values))}}
}

void test_builtin_clzg(short s, int i, unsigned int ui, __int128 i128,
                       _BitInt(128) bi128, double d, int2 i2) {
  __builtin_clzg();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_clzg(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected at most 2, have 3}}
  __builtin_clzg(s);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'short')}}
  __builtin_clzg(i);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int')}}
  __builtin_clzg(i128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '__int128')}}
  __builtin_clzg(bi128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '_BitInt(128)')}}
  __builtin_clzg(d);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'double')}}
  __builtin_clzg(i2);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int2' (vector of 2 'int' values))}}
  __builtin_clzg(i2);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int2' (vector of 2 'int' values))}}
  __builtin_clzg(ui, ui);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'unsigned int')}}
  __builtin_clzg(ui, i128);
  // expected-error@-1 {{2nd argument must be an 'int' (was '__int128')}}
  __builtin_clzg(ui, bi128);
  // expected-error@-1 {{2nd argument must be an 'int' (was '_BitInt(128)')}}
  __builtin_clzg(ui, d);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'double')}}
  __builtin_clzg(ui, i2);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'int2' (vector of 2 'int' values))}}
}

void test_builtin_ctzg(short s, int i, unsigned int ui, __int128 i128,
                       _BitInt(128) bi128, double d, int2 i2) {
  __builtin_ctzg();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_ctzg(i, i, i);
  // expected-error@-1 {{too many arguments to function call, expected at most 2, have 3}}
  __builtin_ctzg(s);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'short')}}
  __builtin_ctzg(i);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int')}}
  __builtin_ctzg(i128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '__int128')}}
  __builtin_ctzg(bi128);
  // expected-error@-1 {{1st argument must be an unsigned integer (was '_BitInt(128)')}}
  __builtin_ctzg(d);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'double')}}
  __builtin_ctzg(i2);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int2' (vector of 2 'int' values))}}
  __builtin_ctzg(i2);
  // expected-error@-1 {{1st argument must be an unsigned integer (was 'int2' (vector of 2 'int' values))}}
  __builtin_ctzg(ui, ui);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'unsigned int')}}
  __builtin_ctzg(ui, i128);
  // expected-error@-1 {{2nd argument must be an 'int' (was '__int128')}}
  __builtin_ctzg(ui, bi128);
  // expected-error@-1 {{2nd argument must be an 'int' (was '_BitInt(128)')}}
  __builtin_ctzg(ui, d);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'double')}}
  __builtin_ctzg(ui, i2);
  // expected-error@-1 {{2nd argument must be an 'int' (was 'int2' (vector of 2 'int' values))}}
}
