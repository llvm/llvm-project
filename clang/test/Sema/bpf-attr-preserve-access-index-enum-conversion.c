// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -Wconversion -verify %s
// RUN: %clang_cc1 -triple bpf -fsyntax-only -Wconversion -verify %s
// expected-no-diagnostics

// A relocated enumerator is converted from the unsigned long result of
// __builtin_preserve_enum_value() to its own type. The user does not write
// that conversion, so -Wconversion, e.g., -Wshorten-64-to-32, must not warn
// about it.
enum E { A = 4 } __attribute__((preserve_access_index));
enum L { BIG = 0x100000000L } __attribute__((preserve_access_index));

int cmp(int x) { return x < A; }
int init(void) { int v = A; return v; }
int arith(int x) { return x + A * 2; }
unsigned long big(void) { return BIG; }
