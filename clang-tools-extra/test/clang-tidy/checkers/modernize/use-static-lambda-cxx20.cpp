// RUN: %check_clang_tidy -std=c++20 %s modernize-use-static-lambda %t

// The check is gated on C++23; no warnings should be emitted in C++20.

void noWarnings() {
  auto f1 = [](int x) { return x * x; };
  auto f2 = [] { return 42; };
  (void)f1;
  (void)f2;
}
