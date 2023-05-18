// RUN: %clang_cc1 -std=c++11 -verify %s -Wunused

namespace {
double operator"" _x(long double value) { return double(value); }
int operator"" _ii(long double value) { return int(value); } // expected-warning {{not needed and will not be emitted}}
}

namespace rdar13589856 {
  template<class T> double value() { return 3.2_x; }
  template<class T> int valuei() { return 3.2_ii; }

  double get_value() { return value<double>(); }
}
