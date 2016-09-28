// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fapinotes-cache-path=%t/APINotesCache -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

#import <SomeKit/SomeKit.h>

int main() {
  A *a;

  [a transform: 0 integer: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  [a setNonnullAInstance: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [A setNonnullAInstance: 0]; // no warning
  
  [a setNonnullAClass: 0]; // no warning
  [A setNonnullAClass: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  [a setNonnullABoth: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}
  [A setNonnullABoth: 0]; // expected-warning{{null passed to a callee that requires a non-null argument}}

  return 0;
}

