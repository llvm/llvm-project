// RUN: %clang_cc1 -fsyntax-only -std=c++26 %s -verify=nontemplate
// RUN: %clang_cc1 -fsyntax-only %s -verify=nontemplate,compat

void decompose_array() {
  int arr[4] = {1, 2, 3, 6};
  auto [x, ...rest, y] = arr; // nontemplate-error{{pack declaration outside of template}} \
  // compat-warning{{structured binding pack is incompatible with C++ standards before C++2c}}
}
