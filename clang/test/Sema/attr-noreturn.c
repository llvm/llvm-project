// RUN: %clang_cc1 -verify -fsyntax-only %s

static void (*fp0)(void) __attribute__((noreturn));

void fatal();

static void __attribute__((noreturn)) f0(void) {
  fatal();
} // expected-warning {{function declared 'noreturn' should not return}}

// On K&R
int f1() __attribute__((noreturn));

int g0 __attribute__((noreturn)); // expected-warning {{'noreturn' only applies to function types; type here is 'int'}}

int f2() __attribute__((noreturn(1, 2))); // expected-error {{'noreturn' attribute takes no arguments}}

void f3() __attribute__((noreturn));
void f3() {
  return;  // expected-warning {{function 'f3' declared 'noreturn' should not return}}
}

#pragma clang diagnostic error "-Winvalid-noreturn"

void f4() __attribute__((noreturn));
void f4() {
  return;  // expected-error {{function 'f4' declared 'noreturn' should not return}}
}

// PR4685
extern void f5 (unsigned long) __attribute__ ((__noreturn__));

void
f5 (unsigned long size)
{

}

// PR2461
__attribute__((noreturn)) void f(__attribute__((noreturn)) void (*x)(void)) {
  x();
}

typedef void (*Fun)(void) __attribute__ ((noreturn(2))); // expected-error {{'noreturn' attribute takes no arguments}}


typedef void fn_t(void);

fn_t *fp __attribute__((noreturn));
void __attribute__((noreturn)) f6(int i) {
  fp();
}

fn_t *fps[4] __attribute__((noreturn));
void __attribute__((noreturn)) f7(int i) {
  fps[i]();
}

extern fn_t *ifps[] __attribute__((noreturn));
void __attribute__((noreturn)) f8(int i) {
  ifps[i]();
}

void __attribute__((noreturn)) f9(int n) {
  extern int g9(int, fn_t **);
  fn_t *fp[n] __attribute__((noreturn));
  int i = g9(n, fp);
  fp[i]();
}

typedef fn_t *fptrs_t[4];
fptrs_t ps __attribute__((noreturn));
void __attribute__((noreturn)) f10(int i) {
  ps[i]();
}
