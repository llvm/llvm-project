// RUN: %clang_cc1 -analyze -std=c++11 -analyzer-checker=alpha.clone.CloneChecker -verify %s

// expected-no-diagnostics

void foo1(int a, long b) {
  auto l = [a, b](){};
}

void foo2(int a, long b) {
  auto l = [&a, b](){};
}

void foo3(int a, long b) {
  auto l = [a](){};
}

void foo4(int a, long b) {
  auto l = [=](){};
}

void foo5(int a, long b) {
  auto l = [&](){};
}

