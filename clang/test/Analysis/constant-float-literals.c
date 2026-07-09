// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify \
// RUN:   -analyzer-config eagerly-assume=false %s

void clang_analyzer_dump_float(float);
void clang_analyzer_dump_double(double);
void clang_analyzer_eval(int);

//===----------------------------------------------------------------------===//
// Floating-point literals are modeled as ConcreteFloat SVals.
//===----------------------------------------------------------------------===//

void testFloatLiterals(void) {
  clang_analyzer_dump_float(0.0f);   // expected-warning{{0 IEEEsingle}}
  clang_analyzer_dump_float(1.0f);   // expected-warning{{1 IEEEsingle}}
  clang_analyzer_dump_float(3.14f);  // expected-warning{{3.1400001 IEEEsingle}}
  clang_analyzer_dump_double(0.0);   // expected-warning{{0 IEEEdouble}}
  clang_analyzer_dump_double(1.0);   // expected-warning{{1 IEEEdouble}}
  clang_analyzer_dump_double(3.14);  // expected-warning{{3.1400000000000001 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Variables assigned from literals retain the ConcreteFloat value.
//===----------------------------------------------------------------------===//

void testVariables(void) {
  float f = 1.5f;
  double d = 2.5;
  clang_analyzer_dump_float(f);   // expected-warning{{1.5 IEEEsingle}}
  clang_analyzer_dump_double(d);  // expected-warning{{2.5 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Float-to-integer casts (truncation).
//===----------------------------------------------------------------------===//

void testFloatToInt(void) {
  float f = 1.9f;
  double d = 2.7;
  int i = (int)f;
  int j = (int)d;
  clang_analyzer_eval(i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 2); // expected-warning{{TRUE}}
}

//===----------------------------------------------------------------------===//
// Float-to-bool casts.
//===----------------------------------------------------------------------===//

void testFloatToBool(void) {
  float zero = 0.0f;
  float nonzero = 1.0f;
  clang_analyzer_eval((int)((_Bool)zero) == 0);    // expected-warning{{TRUE}}
  clang_analyzer_eval((int)((_Bool)nonzero) == 1); // expected-warning{{TRUE}}
}

//===----------------------------------------------------------------------===//
// Float-to-float casts (precision change without loss).
//===----------------------------------------------------------------------===//

void testFloatUpcast(void) {
  float f = 1.5f;
  double d = f;
  // 1.5 is exactly representable in both, so no loss.
  clang_analyzer_dump_double(d); // expected-warning{{1.5 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Unknown float values (parameters, arithmetic results).
//===----------------------------------------------------------------------===//

void testUnknown(float f) {
  clang_analyzer_dump_float(f);        // expected-warning{{Unknown}}
  clang_analyzer_dump_float(f + 1.0f); // expected-warning{{Unknown}}
}

//===----------------------------------------------------------------------===//
// Division by zero detection with concrete floats.
//===----------------------------------------------------------------------===//

float testDivByZeroFloat(void) {
  float x = 0.0f;
  return 1.0f / x; // expected-warning{{Division by zero}}
}
