// Test lines unreachable from C++. Particularly when floats are used in a
// boolean context (C++ inserts implicit FloatingToBoolean conversions,
// C compares to 0).
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump_float(float);
void clang_analyzer_eval(int);

void testBranchOnFloat(void) {
  float nonzero = 1.5f;
  if (nonzero)
    clang_analyzer_dump_float(nonzero); // expected-warning{{1.5 IEEEsingle}}
  else
    clang_analyzer_dump_float(nonzero);

  float zero = 0.0f;
  if (zero)
    clang_analyzer_dump_float(zero);
  else
    clang_analyzer_dump_float(zero);  // expected-warning{{0 IEEEsingle}}

  float negzero = -0.0f;
  if (negzero)
    clang_analyzer_dump_float(negzero);
  else
    clang_analyzer_dump_float(negzero); // expected-warning{{-0 IEEEsingle}}
}

void testLogicalNotOnFloat(void) {
  float nonzero = 0.5f, zero = 0.0f, negzero = -0.0f;
  clang_analyzer_eval(!nonzero);  // expected-warning{{FALSE}}
  clang_analyzer_eval(!zero);     // expected-warning{{TRUE}}
  clang_analyzer_eval(!negzero);  // expected-warning{{TRUE}}
}

void testUnknownFloatCondition(float f) {
  if (f)
    clang_analyzer_dump_float(f); // expected-warning{{Unknown}}
  else
    clang_analyzer_dump_float(f); // expected-warning{{Unknown}}
}
