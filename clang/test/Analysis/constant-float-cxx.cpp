// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_dump_float(float);
void clang_analyzer_dump_double(double);
void clang_analyzer_eval(int);

template <typename T> T twice(T x) { return x + x; }

struct Vec {
  float v;
  Vec operator+(Vec o) const { return Vec{v + o.v}; }
};

constexpr double half(double x) { return x / 2.0; }

double defaultArg(double i = 42) { return i; }
float narrowingDefaultArg(float i = 1.5) { return i; }

void testTemplate() {
  clang_analyzer_dump_float(twice(1.5f)); // expected-warning{{3 IEEEsingle}}
  clang_analyzer_dump_double(twice(2.5)); // expected-warning{{5 IEEEdouble}}
}

void testOverloadedOperator() {
  Vec a{1.5f}, b{2.5f};
  clang_analyzer_dump_float((a + b).v); // expected-warning{{4 IEEEsingle}}
}

void testConstexpr() {
  clang_analyzer_dump_double(half(3.0));  // expected-warning{{1.5 IEEEdouble}}
  constexpr double c = 1.25;
  clang_analyzer_dump_double(c);          // expected-warning{{1.25 IEEEdouble}}
}

// A default argument is not evaluated through the CFG, rather it is folded by
// getConstantVal.
void testDefaultArgument() {
  clang_analyzer_dump_double(defaultArg());         // expected-warning{{42 IEEEdouble}}
  clang_analyzer_dump_float(narrowingDefaultArg()); // expected-warning{{1.5 IEEEsingle}}
}

// A negative value does not fit an unsigned integer; the conversion should be
// unmodeled.
void testUnsignedFromNegative() {
  clang_analyzer_eval((unsigned)-1.5f == 0);  // expected-warning{{UNKNOWN}}
}
