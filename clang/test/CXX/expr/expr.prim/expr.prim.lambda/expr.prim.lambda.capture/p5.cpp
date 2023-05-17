// RUN: %clang_cc1 -std=c++20 -verify %s

void f() {
  int x = 0;
  auto g = [x](int x) { return 0; }; // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}} \
                                     // expected-note {{variable 'x' is explicitly captured here}}
  auto h = [y = 0]<typename y>(y) { return 0; };  // expected-error {{declaration of 'y' shadows template parameter}} \
                                                  // expected-note {{template parameter is declared here}}

}
