// RUN: %clang_cc1 -fsyntax-only -Wignored-attributes -verify -std=c23 %s
// RUN: %clang_cc1 -fsyntax-only -Wignored-attributes -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump %s | FileCheck %s

inline int frob(int x) { return x; }

[[deprecated]] int frob(int); // expected-note 2 {{'frob' has been explicitly marked deprecated here}}

void use1() {
  // Using this should give a deprecation warning, but not a nodiscard warning.	
  frob(0); // expected-warning {{'frob' is deprecated}}
}

[[nodiscard]] int frob(int);

void use2() {
  // This should give both warnings.
  frob(0); // expected-warning {{'frob' is deprecated}} \
              expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

[[maybe_unused]] int frob(int);

// Currently, this is only allowed for the standard spelling of the attributes.
void blob() {}                           // expected-note {{previous definition is here}}
__attribute__((deprecated)) void blob(); // expected-warning {{attribute declaration must precede definition}}

// CHECK: FunctionDecl {{.*}} frob

// CHECK: FunctionDecl {{.*}} prev {{.*}} frob
// CHECK:  DeprecatedAttr

// CHECK: FunctionDecl {{.*}} prev {{.*}} frob
// CHECK:  DeprecatedAttr
// CHECK:  WarnUnusedResultAttr

// CHECK: FunctionDecl {{.*}} prev {{.*}} frob
// CHECK:  DeprecatedAttr
// CHECK:  WarnUnusedResultAttr
// CHECK:  UnusedAttr

