// RUN: %clang_cc1 -triple thumbv7-windows -fms-compatibility -Wno-everything -Wimplicit-function-declaration -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armv7-eabi -Wno-everything -Wimplicit-function-declaration -fsyntax-only -verify %s

void check__dmb(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __dmb(0);
}

void check__dsb(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __dsb(0);
}

void check__isb(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __isb(0);
}

void check__yield(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __yield();
}

void check__wfe(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __wfe();
}

void check__wfi(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __wfi();
}

void check__sev(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __sev();
}

void check__sevl(void) {
  // expected-warning@+2{{call to undeclared library function}}
  // expected-note@+1{{include the header <arm_acle.h> or explicitly provide a declaration for}}
  __sevl();
}
