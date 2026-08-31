// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 \
// RUN:   -analyzer-checker=core,debug.ExprInspection -verify %s

template <typename T> void clang_analyzer_dump(T);
void clang_analyzer_eval(bool);

template <typename T> T twice(T x) { return x + x; }

struct Container {
  float v;
  Container operator+(Container o) const { return Container{v + o.v}; }
};

constexpr double half(double x) { return x / 2.0; }

double defaultArg(double i = 42) { return i; }
float narrowingDefaultArg(float i = 1.5) { return i; }

void testTemplate() {
  clang_analyzer_dump(twice(1.5f)); // expected-warning{{3 IEEEsingle}}
  clang_analyzer_dump(twice(2.5));  // expected-warning{{5 IEEEdouble}}
}

void testOverloadedOperator() {
  Container a{1.5f}, b{2.5f};
  clang_analyzer_dump((a + b).v); // expected-warning{{4 IEEEsingle}}
}

void testConstexpr() {
  clang_analyzer_dump(half(3.0)); // expected-warning{{1.5 IEEEdouble}}
  constexpr double c = 1.25;
  clang_analyzer_dump(c);         // expected-warning{{1.25 IEEEdouble}}
}

// A default argument is not evaluated through the CFG, rather it is folded by
// getConstantVal.
void testDefaultArgument() {
  clang_analyzer_dump(defaultArg());          // expected-warning{{42 IEEEdouble}}
  clang_analyzer_dump(narrowingDefaultArg()); // expected-warning{{1.5 IEEEsingle}}
}

// Negative float -> int is well-defined ([conv.fpint], C11 6.3.1.4) by
// discarding the fractional part, but only if the integral part can be
// represented in the destination type (otherwise UB).
void testUnsignedFromNegative() {
  clang_analyzer_eval((unsigned)-1.5f == 0);  // expected-warning{{UNKNOWN}}
  clang_analyzer_eval((unsigned)-0.5f == 0);  // expected-warning{{TRUE}}
  clang_analyzer_eval((int)-1.5f == -1);      // expected-warning{{TRUE}}
}
