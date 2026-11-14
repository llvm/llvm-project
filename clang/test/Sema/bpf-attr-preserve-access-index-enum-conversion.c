// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -Wconversion -verify %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -Wconversion -verify %s

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
