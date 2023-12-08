// RUN: %clang_cc1 -std=c++11 -verify %s

template<int &...Ns> int f() {
  return sizeof...(Ns);
}
template int f<>();

template<typename ...T> int g() {
  return [...x = T()] { // expected-warning 2{{extension}}
    return sizeof...(x);
  }();
}
template int g<>();
