// RUN: %clang_cc1 -std=c++98 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -verify
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

// RUN: %clang_cc1 -std=c++98 -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -verify
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

#ifndef HEADER
#define HEADER

#define NULL 0
template<typename T>
void *f() {
  void *p;  // @15
  return p; // @16
}
#undef NULL
template<typename T>
void *g() {
  void *p;  // @21
  return p; // @22
}

#else

// expected-warning@16 {{uninitialized}}
// expected-note@15 {{initialize}}
// CHECK: fix-it:"{{.*}}":{15:10-15:10}:" = NULL"

// expected-warning@22 {{uninitialized}}
// expected-note@21 {{initialize}}
// CHECK: fix-it:"{{.*}}":{21:10-21:10}:" = 0"

int main() {
  f<int>(); // expected-note {{instantiation}}
  g<int>(); // expected-note {{instantiation}}
}

#endif
