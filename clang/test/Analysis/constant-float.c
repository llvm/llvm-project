// Disable -Wliteral-range since we intentionally induce inf.
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection -Wno-literal-range \
// RUN:   -analyzer-config eagerly-assume=false -verify %s
//
// Only exact results are modeled so the analyzer's behavior should not change
// under a dynamic rounding mode.
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -frounding-math \
// RUN:   -analyzer-checker=core,debug.ExprInspection -Wno-literal-range \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_dump_float(float);
void clang_analyzer_dump_double(double);
void clang_analyzer_dump_int(int);
void clang_analyzer_eval(int);

void testLiterals(void) {
  clang_analyzer_dump_float(0.0f);  // expected-warning{{0 IEEEsingle}}
  clang_analyzer_dump_float(3.14f); // expected-warning{{3.1400001 IEEEsingle}}
  clang_analyzer_dump_double(3.14); // expected-warning{{3.1400000000000001 IEEEdouble}}
}

void testVariables(void) {
  float f = 1.5f;
  clang_analyzer_dump_float(f); // expected-warning{{1.5 IEEEsingle}}
}

void testUnknown(float f) {
  clang_analyzer_dump_float(f);         // expected-warning{{Unknown}}
  clang_analyzer_dump_float(f + 1.0f);  // expected-warning{{Unknown}}
}

// Exactly representable results are independent of rounding mode and
// evaluation precision.
void testArithmetic(void) {
  clang_analyzer_dump_float(1.0f + 2.0f); // expected-warning{{3 IEEEsingle}}
  clang_analyzer_dump_float(5.0f - 1.5f); // expected-warning{{3.5 IEEEsingle}}
  clang_analyzer_dump_float(1.5f * 2.0f); // expected-warning{{3 IEEEsingle}}
  clang_analyzer_dump_float(3.0f / 4.0f); // expected-warning{{0.75 IEEEsingle}}
  clang_analyzer_dump_double(0.5 + 0.25); // expected-warning{{0.75 IEEEdouble}}
  clang_analyzer_dump_float(0.1f + 0.2f); // expected-warning{{Unknown}}
  clang_analyzer_dump_float(1.0f / 3.0f); // expected-warning{{Unknown}}
}

// Comparisons are exact for every value, so all six predicates should fold.
void testComparisons(void) {
  clang_analyzer_eval(1.0f < 2.0f);   // expected-warning{{TRUE}}
  clang_analyzer_eval(1.0f > 2.0f);   // expected-warning{{FALSE}}
  clang_analyzer_eval(2.0f <= 2.0f);  // expected-warning{{TRUE}}
  clang_analyzer_eval(2.0f >= 2.0f);  // expected-warning{{TRUE}}
  clang_analyzer_eval(1.5 == 1.5);    // expected-warning{{TRUE}}
  clang_analyzer_eval(1.5 != 2.5);    // expected-warning{{TRUE}}
}

// Unary negation can be modeled because of sign-bit.
void testNegation(void) {
  clang_analyzer_dump_float(-1.5f);     // expected-warning{{-1.5 IEEEsingle}}
  clang_analyzer_dump_float(-(-1.5f));  // expected-warning{{1.5 IEEEsingle}}
  clang_analyzer_dump_float(-0.0f);     // expected-warning{{-0 IEEEsingle}}
  clang_analyzer_eval(0.0f == -0.0f);   // expected-warning{{TRUE}}
}

// Conversions between floating points should be modeled only when they are
// exact. 3.14 stored in a double sets mantissa bits a float cannot hold, and
// rounding direction can change at runtime.
void testFloatConversions(void) {
  clang_analyzer_dump_double((double)1.5f); // expected-warning{{1.5 IEEEdouble}}
  clang_analyzer_dump_float((float)3.14);   // expected-warning{{Unknown}}
}

// Type punning reinterprets the bits, whereas we only model conversions of the
// value, so decline it in both directions.
void testTypePunning(void) {
  float f = 1.5f;
  clang_analyzer_dump_int(*(int *)&f);     // expected-warning{{Unknown}}
  int i = 5;
  clang_analyzer_dump_float(*(float *)&i); // expected-warning{{Unknown}}
}

// Casts to bool is defined as a comparison to zero.
void testFloatToBool(void) {
  clang_analyzer_eval((_Bool)0.5f);   // expected-warning{{TRUE}}
  clang_analyzer_eval((_Bool)2.0f);   // expected-warning{{TRUE}}
  clang_analyzer_eval((_Bool)-3.0f);  // expected-warning{{TRUE}}
  clang_analyzer_eval((_Bool)0.0f);   // expected-warning{{FALSE}}
  clang_analyzer_eval((_Bool)-0.0f);  // expected-warning{{FALSE}}
}

// !E is defined as (0 == E) with the zero converted to the type of expr. E.
void testLogicalNot(void) {
  float nonzero = 0.5f, zero = 0.0f, negzero = -0.0f;
  clang_analyzer_eval(!nonzero);  // expected-warning{{FALSE}}
  clang_analyzer_eval(!zero);     // expected-warning{{TRUE}}
  clang_analyzer_eval(!negzero);  // expected-warning{{TRUE}}
}

// Casts to integers discard fractional bits, which should be unmodeled when
// the result is out of range.
void testFloatToInt(void) {
  clang_analyzer_eval((int)1.9f == 1);          // expected-warning{{TRUE}}
  clang_analyzer_eval((int)-1.9 == -1);         // expected-warning{{TRUE}}
  clang_analyzer_dump_float((float)(int)1e30f); // expected-warning{{Unknown}}
}

// Infinities and NaNs should not be modeled.
void testNonFiniteIsUnmodeled(void) {
  clang_analyzer_dump_float(1e400f);                // expected-warning{{Unknown}}
  clang_analyzer_dump_float((float)1e300);          // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__FLT_MAX__ * 2.0f);    // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__builtin_inff());      // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__builtin_huge_valf()); // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__builtin_nanf(""));    // expected-warning{{Unknown}}
  clang_analyzer_dump_double(0.0 / 0.0);            // expected-warning{{Unknown}}
  clang_analyzer_dump_double(1.0 / 0.0);            // expected-warning{{Unknown}}
}

void testSelfArithmetic(void) {
  float f = 1.5f;
  clang_analyzer_dump_float(f - f); // expected-warning{{0 IEEEsingle}}
  clang_analyzer_eval(f == f);      // expected-warning{{TRUE}}
}

// Subnormals should not be modeled.
void testSubnormals(void) {
  clang_analyzer_dump_float(__FLT_DENORM_MIN__);         // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__FLT_MIN__ / 2.0f);         // expected-warning{{Unknown}}
  clang_analyzer_dump_float(__FLT_MIN__ * __FLT_MIN__);  // expected-warning{{Unknown}}
  clang_analyzer_dump_float((float)(double)__FLT_DENORM_MIN__);
  // expected-warning@-1{{Unknown}}
  clang_analyzer_dump_float(__FLT_MIN__);
  // expected-warning@-1{{1.17549435E-38 IEEEsingle}}
}

// Integer to float conversions should only be modeled when the conversion is
// exact.
void testIntToFloat(void) {
  float f = 1;
  clang_analyzer_dump_float(f);                // expected-warning{{1 IEEEsingle}}
  clang_analyzer_dump_float(1.0f + 1);         // expected-warning{{2 IEEEsingle}}
  clang_analyzer_dump_float((float)-3);        // expected-warning{{-3 IEEEsingle}}
  clang_analyzer_dump_float((float)16777216);  // expected-warning{{16777216 IEEEsingle}}
  clang_analyzer_dump_float((float)16777217);  // expected-warning{{Unknown}}

  // 2^26 can be represented, 2^26 + 1 cannot.
  long yes = 1L << 26;
  long no = yes + 1L;
  clang_analyzer_dump_float((float)yes);  // expected-warning{{67108864 IEEEsingle}}
  clang_analyzer_dump_float((float)no);   // expected-warning{{Unknown}}
}

// Complex floating-point types are not modeled.
void testComplexIsUnmodeled(void) {
  _Complex float z = 1.5f;
  clang_analyzer_dump_float(__real__ z); // expected-warning{{Unknown}}
  clang_analyzer_eval(!z);               // expected-warning{{UNKNOWN}}
}

// Inc/decrement operators compute through the same path as += and -=
// respectively in the analyzer.
void testIncrementDecrement(void) {
  float g = 2.0f;
  g += 1.0f;
  clang_analyzer_dump_float(g);   // expected-warning{{3 IEEEsingle}}
  --g;
  clang_analyzer_dump_float(g);   // expected-warning{{2 IEEEsingle}}
  clang_analyzer_dump_float(g++); // expected-warning{{2 IEEEsingle}}
  clang_analyzer_dump_float(g);   // expected-warning{{3 IEEEsingle}}

  float small = 0.5f;
  ++small;
  clang_analyzer_dump_float(small); // expected-warning{{1.5 IEEEsingle}}

  float inexact = 0.1f;
  ++inexact;
  clang_analyzer_dump_float(inexact); // expected-warning{{Unknown}}

  float max = __FLT_MAX__;
  ++max;
  clang_analyzer_dump_float(max); // expected-warning{{Unknown}}
}
