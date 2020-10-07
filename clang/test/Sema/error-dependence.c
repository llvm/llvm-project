// RUN: %clang_cc1 -fsyntax-only -verify -frecovery-ast -fno-recovery-ast-type %s

int call(int); // expected-note2 {{'call' declared here}}

void test1(int s) {
  // verify "assigning to 'int' from incompatible type '<dependent type>'" is
  // not emitted.
  s = call(); // expected-error {{too few arguments to function call}}
}

void test2(int* ptr, float f) {
  // verify diagnostic "used type '<dependent type>' where arithmetic or pointer
  // type is required" is not emitted.
  (call() ? ptr : f); // expected-error {{too few arguments to function call}}
}
