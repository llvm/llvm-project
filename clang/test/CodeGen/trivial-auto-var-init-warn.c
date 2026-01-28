// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -Wtrivial-auto-var-init -emit-llvm -o /dev/null -verify=zero %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -Wtrivial-auto-var-init -emit-llvm -o /dev/null -verify=pattern %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o /dev/null -verify=noflag %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -Wno-trivial-auto-var-init -emit-llvm -o /dev/null -verify=suppressed %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -emit-llvm -o /dev/null -verify=default %s

// noflag-no-diagnostics
// suppressed-no-diagnostics
// default-no-diagnostics

void use(int *);

void switch_precase(int c) {
  switch (c) {
    int x; // zero-warning{{'x' cannot be initialized with '-ftrivial-auto-var-init'}}
           // pattern-warning@-1{{'x' cannot be initialized with '-ftrivial-auto-var-init'}}
  case 0:
    x = 1;
    use(&x);
    break;
  }
}

void goto_bypass(void) {
  goto skip;
  int y; // zero-warning{{'y' cannot be initialized with '-ftrivial-auto-var-init'}}
         // pattern-warning@-1{{'y' cannot be initialized with '-ftrivial-auto-var-init'}}
skip:
  use(&y);
}

void normal_var(void) {
  int x; // no warning: reachable code
  use(&x);
}
