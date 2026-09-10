// We can't assume a double will be wider than a float.
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -mdouble=32 \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_dump_float(float);
void clang_analyzer_dump_double(double);
void clang_analyzer_dumpSvalType_double(double);
void clang_analyzer_eval(int);

void testDoubleIsSingle(void) {
  clang_analyzer_dump_double(3.14);         // expected-warning{{3.1400001 IEEEsingle}}
  clang_analyzer_dumpSvalType_double(3.14); // expected-warning{{float}}
}

void testConversionsAreExact(void) {
  clang_analyzer_dump_float((float)3.14);   // expected-warning{{3.1400001 IEEEsingle}}
  clang_analyzer_dump_double((double)1.5f); // expected-warning{{1.5 IEEEsingle}}
  clang_analyzer_eval(3.14 == 3.14f);       // expected-warning{{TRUE}}
}

void testInexactStillUnmodeled(void) {
  clang_analyzer_dump_double(0.1 + 0.2);  // expected-warning{{Unknown}}
}
