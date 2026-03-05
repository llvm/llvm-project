// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Function reference parameter with multiple pointer indirections
//
// When passing a reference to a function type as a parameter, where the function
// signature contains multiple levels of pointer indirection in its parameters,
// CIR fails during type lowering or function call code generation.

const char *a;
unsigned b;
unsigned char c;
void d(int (&e)(unsigned char *, unsigned *, char, const char **)) {
  e(&c, &b, 0, &a);
}
