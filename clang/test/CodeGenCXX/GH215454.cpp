// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -verify -emit-llvm -o - %s | FileCheck %s

void foo();
void bar();

// A declaration that declares nothing as the substatement dropped the whole
// 'if', including the call in its condition.
// CHECK-LABEL: define{{.*}} void @_Z1di(
// CHECK: call void @_Z3foov()
void d(int e) {
  if (foo(), e) int; // expected-warning {{declaration does not declare anything}} \
                     // expected-warning {{if statement has empty body}} \
                     // expected-note {{put the semicolon on a separate line to silence this warning}}
}

// The empty declaration is the body; the next statement must not become it.
// CHECK-LABEL: define{{.*}} void @_Z1fi(
// CHECK: if.then:
// CHECK-NEXT: br label %if.end
// CHECK: if.end:
// CHECK-NEXT: call void @_Z3barv()
void f(int e) {
  if (e) int; // expected-warning {{declaration does not declare anything}} \
              // expected-warning {{if statement has empty body}} \
              // expected-note {{put the semicolon on a separate line to silence this warning}}
  bar();
}
