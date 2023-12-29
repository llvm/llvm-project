// RUN: %clang_cc1 -verify -std=c++20 -Wall %s
// RUN: cp %s %t
// RUN: %clang_cc1 -x c++ -std=c++20 -fixit %t
// RUN: %clang_cc1 -Wall -Werror -x c++ -std=c++20 %t
// RUN: cat %t | FileCheck %s

namespace std {

int &&move(auto &&a) { return a; }

int &&forward(auto &a) { return a; }

} // namespace std

namespace mystd {

[[clang::behaves_like_std("move")]] int &&move(auto &&a) { return a; }

[[clang::behaves_like_std("forward")]] int &&forward(auto &a) { return a; }

} // namespace mystd

[[clang::behaves_like_std("move")]] int &&mymove(auto &&a) { return a; }

[[clang::behaves_like_std("forward")]] int &&myforward(auto &a) { return a; }

void f() {
  using namespace std;
  int i = 0;
  (void)move(i); // expected-warning {{unqualified call to 'std::move}}
  // CHECK: {{^}}  (void)std::move
  (void)forward(i); // expected-warning {{unqualified call to 'std::forward}}
  // CHECK: {{^}}  (void)std::forward
}

void g() {
  using namespace mystd;
  int i = 0;
  (void)move(i); // expected-warning {{unqualified call to 'mystd::move}}
  // CHECK: {{^}}  (void)mystd::move
  (void)forward(i); // expected-warning {{unqualified call to 'mystd::forward}}
  // CHECK: {{^}}  (void)mystd::forward
}

void h() {
    int i = 0;
    (void)mymove(i); // no-warning
    (void)myforward(i); // no-warning
}
