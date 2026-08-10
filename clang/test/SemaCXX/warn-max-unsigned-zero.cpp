// RUN: %clang_cc1 -fsyntax-only -Wmax-unsigned-zero -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -Wmax-unsigned-zero %s -std=c++11 -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

namespace std {
template <typename T>
T max(const T &, const T &);
}

void test(unsigned u) {
  auto a = std::max(55u, 0u);
  // expected-warning@-1{{taking the max of a value and unsigned zero is always equal to the other value}}
  // expected-note@-2{{remove call to max function and unsigned zero argument}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:24-[[@LINE-4]]:28}:""
  auto b = std::max(u, 0u);
  // expected-warning@-1{{taking the max of a value and unsigned zero is always equal to the other value}}
  // expected-note@-2{{remove call to max function and unsigned zero argument}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:26}:""
  auto c = std::max(0u, 55u);
  // expected-warning@-1{{taking the max of unsigned zero and a value is always equal to the other value}}
  // expected-note@-2{{remove call to max function and unsigned zero argument}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:24}:""
  auto d = std::max(0u, u);
  // expected-warning@-1{{taking the max of unsigned zero and a value is always equal to the other value}}
  // expected-note@-2{{remove call to max function and unsigned zero argument}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:20}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:21-[[@LINE-4]]:24}:""
}

void negative_test(signed s) {
  auto a = std::max(0, s);
  auto b = std::max(s, 0);
  auto c = std::max(22, 0);
  auto d = std::max(0, 22);
}

template <unsigned x>
unsigned template_test() {
  return std::max(x, 0u);
  // expected-warning@-1{{taking the max of a value and unsigned zero is always equal to the other value}}
  // expected-note@-2{{remove call to max function and unsigned zero argument}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:18}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:20-[[@LINE-4]]:24}:""
}

int a = template_test<0>() + template_test<1>() + template_test<2>();

#define comp(x,y) std::max(x, y)

int b = comp(0, 1);
int c = comp(0u, 1u);
int d = comp(2u, 0u);

