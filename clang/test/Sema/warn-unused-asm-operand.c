// RUN: %clang_cc1 -fsyntax-only -verify=silent %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-asm-operand -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wasm -verify %s

// silent-no-diagnostics

int add(int a, int b) {
  int r;
  // All operands are referenced: no warning.
  asm("add %1, %2, %0" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

int named(int x) {
  int r;
  // Named operands are still tracked correctly: no warning.
  asm("mov %[in], %[out]" : [out] "=r"(r) : [in] "r"(x));
  return r;
}

int tied(int a) {
  int r;
  // A numerically-tied input ("0") is referenced via the output's own
  // number by convention, not its own: no warning for the input.
  asm("inc %0" : "=r"(r) : "0"(a));
  return r;
}

void readwrite(int *p) {
  // A read-write operand is a single slot referenced via %0: no warning.
  asm("incl %0" : "+r"(*p));
}

int unused_output(int a) {
  int r;
  // Neither operand is referenced by the template: both warn.
  // expected-warning@+2 {{unused asm output operand}}
  // expected-warning@+1 {{unused asm input operand}}
  asm("nop" : "=r"(r) : "r"(a));
  return a;
}

int unused_input(int a, int b) {
  int r;
  // expected-warning@+1 {{unused asm input operand}}
  asm("add %1, %1, %0" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

int unused_alongside_tied(int a, int b) {
  int r;
  // The tied operand ("0") stays silent; only the genuinely-unreferenced
  // one warns.
  // expected-warning@+1 {{unused asm input operand}}
  asm("inc %0" : "=r"(r) : "0"(a), "r"(b));
  return r;
}
