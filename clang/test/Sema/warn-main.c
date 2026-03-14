// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -x c++ %s 2>&1 | FileCheck %s

// expected-note@+1 2{{previous definition is here}}
int main(void) {
  return 0;
}

// expected-error@+2 {{static declaration of 'main' follows non-static declaration}}
// expected-warning@+1 {{'main' should not be declared static}}
static int main(void) {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:8}:""
  return 0;
}

// expected-error@+2 {{redefinition of 'main'}}
// expected-error@+1 {{'main' is not allowed to be declared inline}}
inline int main(void) {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:8}:""
  return 0;
}

// expected-warning@+5 {{function 'main' declared 'noreturn' should not return}}
// expected-warning@+2 {{'main' is not allowed to be declared _Noreturn}}
// expected-note@+1 {{remove '_Noreturn'}}
_Noreturn int main(void) {
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:1-[[@LINE-1]]:11}:""
  return 0;
}

// expected-warning@+1 {{'main' is not allowed to be declared variadic}}
int main(int argc, char**argv, ...) { return 0; }
