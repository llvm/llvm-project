// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
constexpr int
midpoint(int __a, int ) {
  constexpr unsigned  __half_diff = 0;
  return __half_diff;
}
}
struct Tuple {
  int min;
  int mid;
  constexpr Tuple() {
    min = 0;
    mid = std::midpoint(min, min);
  }
};
constexpr Tuple tup;

