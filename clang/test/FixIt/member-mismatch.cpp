// RUN: %clang_cc1 -verify %s
// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

class Foo { // #defined-here
  int get() const; // expected-note {{because it is const qualified}}
  void set(int); // expected-note {{because it is not const qualified}}
};

// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:15-[[@LINE+1]]:15}:" const"
int Foo::get() {} // expected-error {{does not match any declaration}}
                  // expected-note@#defined-here {{defined here}}
// CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:20-[[@LINE+1]]:26}:""
void Foo::set(int) const {} // expected-error {{does not match any declaration}}
                            // expected-note@#defined-here {{defined here}}
