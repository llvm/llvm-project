// RUN: %clang_cc1 -verify -x c++ -std=c++11 %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -x c++ -std=c++11 %s 2>&1 | FileCheck %s

struct X {
  int i;
};

void func() {
  // CHECK: fix-it:"{{.*}}":{10:6-10:8}:"{}"
  X x(); // expected-warning {{function declaration}} expected-note{{replace parentheses with an initializer}}
  
  typedef int *Ptr;
  // CHECK: fix-it:"{{.*}}":{14:8-14:10}:" = nullptr"
  Ptr p(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

  // CHECK: fix-it:"{{.*}}":{17:15-17:17}:" = u'\\0'"
  char16_t u16(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

  // CHECK: fix-it:"{{.*}}":{20:15-20:17}:" = U'\\0'"
  char32_t u32(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}
}
