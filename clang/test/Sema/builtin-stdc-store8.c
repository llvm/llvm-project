// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2y -isystem %S/Inputs -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
#include <stdbit.h>

// Null pointer is diagnosed (NonNull attribute).
void test_null(void) {
  stdc_store8_leu8(1, 0);       // expected-warning{{null passed to a callee that requires a non-null argument}}
  stdc_store8_beu32(1, nullptr); // expected-warning{{null passed to a callee that requires a non-null argument}}
  stdc_store8_aligned_les64(1, 0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

// Only the destination pointer is non-null; a zero value is fine.
void test_zero_value(unsigned char *p) {
  stdc_store8_leu8(0, p);
  stdc_store8_beu32(0, p);
  stdc_store8_aligned_les64(0, p);
}

// Wrong pointer types are rejected by the type system at the call site.
void test_wrong_types(void) {
  int int_arr[1] = {0};
  unsigned int uint_arr[1] = {0};
  char char_arr[4] = "abc";

  stdc_store8_leu32(1, int_arr);  // expected-error{{incompatible pointer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
  stdc_store8_leu32(1, uint_arr); // expected-error{{incompatible pointer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
  stdc_store8_leu32(1, char_arr); // expected-warning{{converts between pointers to integer types}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
}

// The destination is written to, so it must not be const.
void test_const_destination(const unsigned char *cp) {
  const unsigned char arr[4] = {0};
  stdc_store8_leu32(1, cp);  // expected-warning{{discards qualifiers}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
  stdc_store8_leu32(1, arr); // expected-warning{{discards qualifiers}} expected-note@Inputs/stdbit.h:*{{passing argument to parameter here}}
}

// Argument count is checked.
void test_arg_count(unsigned char *p) {
  stdc_store8_leu32(1);       // expected-error{{too few arguments to function call}}
  stdc_store8_leu32(1, p, p); // expected-error{{too many arguments to function call}}
}

// The functions return void.
void test_return_type(unsigned char *p) {
  __typeof__(stdc_store8_leu8(1, p)) *vp = 0;
  void *check = vp;
  (void)check;
  int x = stdc_store8_leu8(1, p); // expected-error{{initializing 'int' with an expression of incompatible type 'void'}}
}

// Every width, signedness, byte order and alignment variant is accepted.
void test_all_variants(unsigned char *p) {
  alignas(8) unsigned char buf[8];
  stdc_store8_leu8(1, p);
  stdc_store8_les8(1, p);
  stdc_store8_beu8(1, p);
  stdc_store8_bes8(1, p);
  stdc_store8_leu16(1, p);
  stdc_store8_les16(1, p);
  stdc_store8_beu16(1, p);
  stdc_store8_bes16(1, p);
  stdc_store8_leu32(1, p);
  stdc_store8_les32(1, p);
  stdc_store8_beu32(1, p);
  stdc_store8_bes32(1, p);
  stdc_store8_leu64(1, p);
  stdc_store8_les64(1, p);
  stdc_store8_beu64(1, p);
  stdc_store8_bes64(1, p);
  stdc_store8_aligned_leu8(1, buf);
  stdc_store8_aligned_les8(1, buf);
  stdc_store8_aligned_beu8(1, buf);
  stdc_store8_aligned_bes8(1, buf);
  stdc_store8_aligned_leu16(1, buf);
  stdc_store8_aligned_les16(1, buf);
  stdc_store8_aligned_beu16(1, buf);
  stdc_store8_aligned_bes16(1, buf);
  stdc_store8_aligned_leu32(1, buf);
  stdc_store8_aligned_les32(1, buf);
  stdc_store8_aligned_beu32(1, buf);
  stdc_store8_aligned_bes32(1, buf);
  stdc_store8_aligned_leu64(1, buf);
  stdc_store8_aligned_les64(1, buf);
  stdc_store8_aligned_beu64(1, buf);
  stdc_store8_aligned_bes64(1, buf);
}

void test_all_builtin_variants(unsigned char *p) {
  alignas(8) unsigned char buf[8];
  __builtin_stdc_store8_leu8(1, p);
  __builtin_stdc_store8_les8(1, p);
  __builtin_stdc_store8_beu8(1, p);
  __builtin_stdc_store8_bes8(1, p);
  __builtin_stdc_store8_leu16(1, p);
  __builtin_stdc_store8_les16(1, p);
  __builtin_stdc_store8_beu16(1, p);
  __builtin_stdc_store8_bes16(1, p);
  __builtin_stdc_store8_leu32(1, p);
  __builtin_stdc_store8_les32(1, p);
  __builtin_stdc_store8_beu32(1, p);
  __builtin_stdc_store8_bes32(1, p);
  __builtin_stdc_store8_leu64(1, p);
  __builtin_stdc_store8_les64(1, p);
  __builtin_stdc_store8_beu64(1, p);
  __builtin_stdc_store8_bes64(1, p);
  __builtin_stdc_store8_aligned_leu8(1, buf);
  __builtin_stdc_store8_aligned_les8(1, buf);
  __builtin_stdc_store8_aligned_beu8(1, buf);
  __builtin_stdc_store8_aligned_bes8(1, buf);
  __builtin_stdc_store8_aligned_leu16(1, buf);
  __builtin_stdc_store8_aligned_les16(1, buf);
  __builtin_stdc_store8_aligned_beu16(1, buf);
  __builtin_stdc_store8_aligned_bes16(1, buf);
  __builtin_stdc_store8_aligned_leu32(1, buf);
  __builtin_stdc_store8_aligned_les32(1, buf);
  __builtin_stdc_store8_aligned_beu32(1, buf);
  __builtin_stdc_store8_aligned_bes32(1, buf);
  __builtin_stdc_store8_aligned_leu64(1, buf);
  __builtin_stdc_store8_aligned_les64(1, buf);
  __builtin_stdc_store8_aligned_beu64(1, buf);
  __builtin_stdc_store8_aligned_bes64(1, buf);
}

// A call cannot be a constant expression in C: it modifies an object.
void test_not_constant(void) {
  unsigned char buf[4];
  static_assert((stdc_store8_leu32(1, buf), 1), ""); // expected-error{{static assertion expression is not an integral constant expression}}
}
