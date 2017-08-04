// RUN: %clang_cc1 -std=c++98 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -verify
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -include-pch %t %s -Wuninitialized -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

#ifndef HEADER
#define HEADER

#define NULL 0
template<typename T>
void *f() {
  void *p;  // @11
  return p; // @12
}
#undef NULL
template<typename T>
void *g() {
  void *p;  // @17
  return p; // @18
}

#else

// expected-warning@12 {{uninitialized}}
// expected-note@11 {{initialize}}
// CHECK: fix-it:"{{.*}}":{11:10-11:10}:" = NULL"

// expected-warning@18 {{uninitialized}}
// expected-note@17 {{initialize}}
// CHECK: fix-it:"{{.*}}":{17:10-17:10}:" = 0"

int main() {
  f<int>(); // expected-note {{instantiation}}
  g<int>(); // expected-note {{instantiation}}
}

#endif
