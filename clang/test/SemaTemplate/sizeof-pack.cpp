// RUN: %clang_cc1 -std=c++11 -verify %s
// expected-no-diagnostics

template<int &...Ns> int f() {
  return sizeof...(Ns);
}
template int f<>();
