// RUN: %clang_cc1 -Wunknown-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wunknown-attributes -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

[[gmu::deprected]] // expected-warning {{unknown attribute 'gmu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f1(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:17}:"gnu::deprecated"

[[gmu::deprecated]] // expected-warning {{unknown attribute 'gmu::deprecated' ignored; did you mean 'gnu::deprecated'?}}
int f2(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:18}:"gnu::deprecated"

[[gnu::deprected]] // expected-warning {{unknown attribute 'gnu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f3(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:17}:"gnu::deprecated"

[[deprected]] // expected-warning {{unknown attribute 'deprected' ignored; did you mean 'deprecated'?}}
int f4(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:12}:"deprecated"

[[using gmu : deprected]] // expected-warning {{unknown attribute 'gmu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f5(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"gnu"
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:24}:"deprecated"

[[using gmu : deprecated]] // expected-warning {{unknown attribute 'gmu::deprecated' ignored; did you mean 'gnu::deprecated'?}}
int f6(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"gnu"

[[using gnu : deprected]] // expected-warning {{unknown attribute 'gnu::deprected' ignored; did you mean 'gnu::deprecated'?}}
int f7(void) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:15-[[@LINE-4]]:24}:"deprecated"

[[using gnu : deprecated, noretyrn]] // expected-warning {{unknown attribute 'gnu::noretyrn' ignored; did you mean 'gnu::noreturn'?}}
void f8(void) {
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:27-[[@LINE-3]]:35}:"noreturn"

[[using gmu : deprected, noretyrn]] // expected-warning {{unknown attribute 'gmu::deprected' ignored; did you mean 'gnu::deprecated'?}} \
                                    // expected-warning {{unknown attribute 'gmu::noretyrn' ignored; did you mean 'gnu::noreturn'?}}
void f9(void) {
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"gnu"
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:24}:"deprecated"

// CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:9-[[@LINE-7]]:12}:"gnu"
// CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:26-[[@LINE-8]]:34}:"noreturn"

__attribute__((cont, deprected)) // expected-warning {{unknown attribute 'cont' ignored; did you mean 'const'?}} \
                                 // expected-warning {{unknown attribute 'deprected' ignored; did you mean 'deprecated'?}}
int f10(int) {
  return 0;
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:16-[[@LINE-5]]:20}:"const"
// CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:22-[[@LINE-6]]:31}:"deprecated"

[[using gnu: noretyrn, address_spaci(0)]] // expected-warning {{unknown attribute 'gnu::noretyrn' ignored; did you mean 'gnu::noreturn'?}} \
                                          // expected-warning {{unknown attribute 'gnu::address_spaci' ignored}}
void f11(void) {
}
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:14-[[@LINE-4]]:22}:"noreturn"
