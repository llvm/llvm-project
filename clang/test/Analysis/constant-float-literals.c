// Semantics of long double differ depending on target, which is why we run on
// multiple targets.
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify=common,x87 %s
// RUN: %clang_analyze_cc1 -triple aarch64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify=common,quad %s
// RUN: %clang_analyze_cc1 -triple x86_64-pc-windows-msvc \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify=common,ldbl64 %s

void clang_analyzer_dump_float(float);
void clang_analyzer_dump_double(double);
void clang_analyzer_dump_longdouble(long double);
void clang_analyzer_eval(int);

//===----------------------------------------------------------------------===//
// Floating-point literals are modeled as ConcreteFloat SVals.
//===----------------------------------------------------------------------===//

void testFloatLiterals(void) {
  clang_analyzer_dump_float(0.0f);  // common-warning{{0 IEEEsingle}}
  clang_analyzer_dump_float(1.0f);  // common-warning{{1 IEEEsingle}}
  clang_analyzer_dump_float(3.14f); // common-warning{{3.1400001 IEEEsingle}}
  clang_analyzer_dump_double(0.0);  // common-warning{{0 IEEEdouble}}
  clang_analyzer_dump_double(1.0);  // common-warning{{1 IEEEdouble}}
  clang_analyzer_dump_double(3.14); // common-warning{{3.1400000000000001 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// long double is modeled with the target's floating-point semantics.
//===----------------------------------------------------------------------===//

void testLongDoubleLiterals(void) {
  // 0.0 and 1.0 are representable exactly in all formats so only semantic name
  // differs on different targets.
  clang_analyzer_dump_longdouble(0.0L); // x87-warning{{0 x87DoubleExtended}}
quad-warning{{0 IEEEquad}} ldbl64-warning{{0 IEEEdouble}}
  clang_analyzer_dump_longdouble(1.0L); // x87-warning{{1 x87DoubleExtended}} quad-warning{{1 IEEEquad}} ldbl64-warning{{1 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Variables assigned from literals retain the ConcreteFloat value.
//===----------------------------------------------------------------------===//

void testVariables(void) {
  float f = 1.5f;
  double d = 2.5;
  clang_analyzer_dump_float(f);   // common-warning{{1.5 IEEEsingle}}
  clang_analyzer_dump_double(d);  // common-warning{{2.5 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Float-to-integer casts (truncation).
//===----------------------------------------------------------------------===//

void testFloatToInt(void) {
  float f = 1.9f;
  double d = 2.7;
  int i = (int)f;
  int j = (int)d;
  clang_analyzer_eval(i == 1);  // common-warning{{TRUE}}
  clang_analyzer_eval(j == 2);  // common-warning{{TRUE}}
}

//===----------------------------------------------------------------------===//
// Float-to-bool casts.
//===----------------------------------------------------------------------===//

void testFloatToBool(void) {
  float zero = 0.0f;
  float nonzero = 1.0f;
  clang_analyzer_eval((int)((_Bool)zero) == 0);     // common-warning{{TRUE}}
  clang_analyzer_eval((int)((_Bool)nonzero) == 1);  // common-warning{{TRUE}}
}

//===----------------------------------------------------------------------===//
// Float-to-float casts (precision change without loss).
//===----------------------------------------------------------------------===//

void testFloatUpcast(void) {
  float f = 1.5f;
  double d = f;
  // 1.5 is exactly representable in both, so no loss.
  clang_analyzer_dump_double(d);  // common-warning{{1.5 IEEEdouble}}
}

//===----------------------------------------------------------------------===//
// Float-to-float casts (inexact narrowing stays Unknown).
//===----------------------------------------------------------------------===//

void testFloatNarrowing(void) {
  double d = 3.14;
  float f = (float)d;
  // 3.14 is not exactly representable in float, and rounding direction is
  // implementation-defined, so we don't model here.
  clang_analyzer_dump_float(f); // common-warning{{Unknown}}
}

//===----------------------------------------------------------------------===//
// Unknown float values (parameters, arithmetic results).
//===----------------------------------------------------------------------===//

void testUnknown(float f) {
  clang_analyzer_dump_float(f);         // common-warning{{Unknown}}
  clang_analyzer_dump_float(f + 1.0f);  // common-warning{{Unknown}}
}

//===----------------------------------------------------------------------===//
// Division by zero detection with concrete floats.
//===----------------------------------------------------------------------===//

float testDivByZeroFloat(void) {
  float x = 0.0f;
  return 1.0f / x;  // common-warning{{Division by zero}}
}
