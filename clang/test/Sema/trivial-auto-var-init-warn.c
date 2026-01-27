// RUN: %clang_cc1 -ftrivial-auto-var-init=zero -Wtrivial-auto-var-init -fsyntax-only -verify=zero %s
// RUN: %clang_cc1 -ftrivial-auto-var-init=pattern -Wtrivial-auto-var-init -fsyntax-only -verify=pattern %s
// RUN: %clang_cc1 -fsyntax-only -verify=noflag %s
// RUN: %clang_cc1 -ftrivial-auto-var-init=zero -Wno-trivial-auto-var-init -fsyntax-only -verify=suppressed %s
// RUN: %clang_cc1 -ftrivial-auto-var-init=zero -fsyntax-only -verify=default %s

// noflag-no-diagnostics
// suppressed-no-diagnostics
// default-no-diagnostics

void use(int *);

void switch_precase(int c) {
  switch (c) {
    int x; // zero-warning{{variable 'x' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
           // pattern-warning@-1{{variable 'x' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
  case 0:
    x = 1;
    use(&x);
    break;
  }
}

void goto_bypass(void) {
  goto skip;
  int y; // zero-warning{{variable 'y' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
         // pattern-warning@-1{{variable 'y' is uninitialized and cannot be initialized with '-ftrivial-auto-var-init' because it is unreachable}}
skip:
  use(&y);
}

void normal_var(void) {
  int x;
  use(&x);
}
