// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -verify -emit-llvm -o - %s -x c++ -DCALL_FOO

#ifndef CALL_FOO
// expected-no-diagnostics
#endif

namespace A {
int foo(void) { // #A-foo
  return 0;
}
} // namespace A

namespace B {
int bar(void) {
  return 1;
}
} // namespace B

namespace C {
int baz(void) {
  return 2;
}
} // namespace C

#pragma omp begin declare variant match(implementation = {vendor(llvm)})

// This will *not* be a specialization of A::foo(void).
int foo(void) { // #global-foo
  return 3;
}

namespace B {
// This will *not* be a specialization of A::foo(void).
int foo(void) {
  return 4;
}
// This will be a specialization of B::bar(void).
int bar(void) {
  return 0;
}
} // namespace B

using namespace C;

// This will be a specialization of C::baz(void).
int baz(void) {
  return 0;
}
#pragma omp end declare variant


int explicit1() {
  // Should return 0.
  return A::foo() + B::bar() + C::baz();
}

int implicit2() {
  using namespace A;
  using namespace B;
  // Should return 0.
#ifdef CALL_FOO
  foo(); // expected-error {{call to 'foo' is ambiguous}}
         //   expected-note@#A-foo {{candidate function}}
         //   expected-note@#global-foo {{candidate function}}
#endif
  return bar() + baz();
}

int main() {
  // Should return 0.
  return explicit1() + implicit2();
}

// CHECK-LABEL: define {{.*}} @_Z9explicit1v
// CHECK:         call {{.*}} @_ZN1A3fooEv
// CHECK:         call {{.*}} @"_ZN1B27bar$ompvariant$S4$s11$PllvmEv"
// CHECK:         call {{.*}} @"_Z27baz$ompvariant$S4$s11$Pllvmv"

// CHECK-LABEL: define {{.*}} @_Z9implicit2v
// CHECK:         call {{.*}} @"_ZN1B27bar$ompvariant$S4$s11$PllvmEv"
// CHECK:         call {{.*}} @"_Z27baz$ompvariant$S4$s11$Pllvmv"
