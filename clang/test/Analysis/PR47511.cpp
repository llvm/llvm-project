// RUN: %clang_analyze_cc1 -std=c++20 -w -analyzer-checker=core -verify %s

// expected-no-diagnostics

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

void test() {
  // No crash 
  (void)(0 <=> 0);
}
