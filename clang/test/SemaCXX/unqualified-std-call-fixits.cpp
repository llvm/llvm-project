// RUN: %clang_cc1 -verify -std=c++20 -Wall %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -std=c++20 -fixit %t
// RUN: %clang_cc1 -Wall -Werror -x c++ -std=c++20 %t
// RUN: cat %t | FileCheck %s

namespace std {

int &&move(auto &&a) { return a; }

int &&forward(auto &a) { return a; }

} // namespace std

using namespace std;

void f() {
  int i = 0;
  (void)move(i); // expected-warning {{unqualified call to 'std::move}}
  // CHECK: {{^}}  (void)std::move
  (void)forward(i); // expected-warning {{unqualified call to 'std::forward}}
  // CHECK: {{^}}  (void)std::forward
}
