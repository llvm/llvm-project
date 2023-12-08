// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

enum class Color { Red, Green, Blue };

struct ConvertsToColorA {
  operator Color();
};

struct ConvertsToColorB {
  operator Color();
};

Color foo(bool cond, ConvertsToColorA ca, ConvertsToColorB cb) {
  return cond? ca : cb;
}
