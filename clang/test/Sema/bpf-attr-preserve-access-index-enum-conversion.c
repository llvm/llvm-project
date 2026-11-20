// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -Wconversion -Wtautological-value-range-compare -verify %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -Wconversion -Wtautological-value-range-compare -verify %s

// A relocated enumerator is converted from the unsigned long result of
// __builtin_preserve_enum_value() to its own type, e.g., int. The user does not
// write that conversion, so -Wconversion, e.g., -Wsign-conversion or
// -Wshorten-64-to-32, must not warn about it.
enum E { A = 4 } __attribute__((preserve_access_index));
enum L { BIG = 0x100000000L } __attribute__((preserve_access_index));
enum U : unsigned char { UC = 4 } __attribute__((preserve_access_index));

unsigned long big(void) { return BIG; }
unsigned long uc(void) { return UC; }

// The enumerator has type int, as without the attribute, so converting it to
// int is not diagnosed, unlike an explicit __builtin_preserve_enum_value() call.
int narrow(void) { int x = A; return x; }
int narrow_builtin(void) {
  return __builtin_preserve_enum_value(*(enum E *)A, 1); // expected-warning {{implicit conversion loses integer precision: 'unsigned long' to 'int'}}
}

// A relocated enumerator is not a constant, but the range of its local value is
// used, as without the attribute, so its conversion to another integer type and
// its comparison with an unsigned value are not diagnosed if the local value
// fits.
enum F { D = 300 } __attribute__((preserve_access_index));

unsigned to_unsigned(void) { unsigned u = A; return u; }
unsigned char to_uchar(void) { return A; }
unsigned char to_uchar_trunc(void) {
  return D; // expected-warning {{implicit conversion loses integer precision: 'int' to 'unsigned char'}}
}
int cmp_unsigned(unsigned x) { return x == A || x < A; }

// The value of a relocated enumerator is only known at load time, so its
// comparison with a constant is not tautological, even if the constant does
// not fit in the bits of its local value.
int cmp_const(void) { return A == 11 || 11 != A; }

// A case value is converted to the promoted type of the switch condition.
int case_unsigned(unsigned x, enum E e, unsigned char c) {
  switch (x) {
  case A:
    return 1;
  }
  switch (e) {
  case A:
    return 2;
  }
  switch (c) {
  case D: // expected-warning {{overflow converting case value to switch condition type (300 to 44)}}
    return 3;
  }
  return 0;
}
