// Multiple targets needed because long double semantics differ on them.
//
// DEFINE: %{analyze}= %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection
//
// RUN: %{analyze} -triple x86_64-unknown-linux-gnu -verify=x87 %s
// RUN: %{analyze} -triple aarch64-unknown-linux-gnu -verify=quad %s
// RUN: %{analyze} -triple x86_64-pc-windows-msvc -verify=ldbl64 %s
// RUN: %{analyze} -triple powerpc64le-unknown-linux-gnu -verify=ibm128 %s

template <typename T> void clang_analyzer_dump(T);
template <typename T> void clang_analyzer_dumpSvalType(T);
void clang_analyzer_eval(bool);

void testVariables(void) {
  long double ld = 1.5L;
  clang_analyzer_dump(ld);
  // x87-warning@-1{{1.5 x87DoubleExtended}}
  // quad-warning@-2{{1.5 IEEEquad}}
  // ldbl64-warning@-3{{1.5 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
}

// long double has different semantics depending on target. IBM double-double
// should not be modeled.
void testLongDouble(void) {
  clang_analyzer_dump(1.0L);
  // x87-warning@-1{{1 x87DoubleExtended}}
  // quad-warning@-2{{1 IEEEquad}}
  // ldbl64-warning@-3{{1 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
  clang_analyzer_dumpSvalType(1.0L);
  // x87-warning@-1{{long double}}
  // quad-warning@-2{{__float128}}
  // ldbl64-warning@-3{{double}}
  // ibm128-warning@-4{{NULL TYPE}}
}

void testLongDoubleArithmetic(void) {
  clang_analyzer_dump(1.0L + 2.0L);
  // x87-warning@-1{{3 x87DoubleExtended}}
  // quad-warning@-2{{3 IEEEquad}}
  // ldbl64-warning@-3{{3 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
  clang_analyzer_dump(3.0L / 4.0L);
  // x87-warning@-1{{0.75 x87DoubleExtended}}
  // quad-warning@-2{{0.75 IEEEquad}}
  // ldbl64-warning@-3{{0.75 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
  clang_analyzer_eval(1.0L < 2.0L);
  // x87-warning@-1{{TRUE}}
  // quad-warning@-2{{TRUE}}
  // ldbl64-warning@-3{{TRUE}}
  // ibm128-warning@-4{{UNKNOWN}}
}

void testIntToFloat(void) {
  clang_analyzer_dump((long double)1);
  // x87-warning@-1{{1 x87DoubleExtended}}
  // quad-warning@-2{{1 IEEEquad}}
  // ldbl64-warning@-3{{1 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
}

// IBM double-double should not be created on the makeZeroVal build path.
void testZeroInitialized(void) {
  static long double s;
  clang_analyzer_dump(s);
  // x87-warning@-1{{0 x87DoubleExtended}}
  // quad-warning@-2{{0 IEEEquad}}
  // ldbl64-warning@-3{{0 IEEEdouble}}
  // ibm128-warning@-4{{Unknown}}
}
