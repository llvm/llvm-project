// _Float16 and __bf16 are both 16 bits wide but have different semantics, so
// the same bit pattern denotes different values in each.
//
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s
// RUN: %clang_analyze_cc1 -triple aarch64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_dump_half(_Float16);
void clang_analyzer_dump_bfloat(__bf16);
void clang_analyzer_dumpSvalType_half(_Float16);
void clang_analyzer_dumpSvalType_bfloat(__bf16);
void clang_analyzer_eval(int);

void test16BitValues(void) {
  clang_analyzer_dump_half((_Float16)1.5f);         // expected-warning{{1.5 IEEEhalf}}
  clang_analyzer_dump_bfloat((__bf16)1.5f);         // expected-warning{{1.5 BFloat}}
  clang_analyzer_dumpSvalType_half((_Float16)1.5f); // expected-warning{{_Float16}}
  clang_analyzer_dumpSvalType_bfloat((__bf16)1.5f); // expected-warning{{__bf16}}
}

// 0x3C00 is 1.0 as an IEEEhalf and 0.0078125 as a BFloat. Ensure conversions
// between them are respected.
void testSameBitsDifferentSemantics(void) {
  clang_analyzer_dump_bfloat((__bf16)0.0078125f); // expected-warning{{0.007813 BFloat}}
  _Float16 a = (_Float16)1.0f;
  clang_analyzer_dump_half(a);                    // expected-warning{{1 IEEEhalf}}
  clang_analyzer_dump_half(a + a);                // expected-warning{{2 IEEEhalf}}
  clang_analyzer_eval(a + a == (_Float16)2.0f);   // expected-warning{{TRUE}}
}

// 1 + 2^-9 needs 10 mantissa bits, which fits an IEEEhalf but not a BFloat, so
// result depends on rounding semantics => unmodeled.
void testExactnessIsPerSemantics(void) {
  clang_analyzer_dump_half((_Float16)1.001953125f); // expected-warning{{1.002 IEEEhalf}}
  clang_analyzer_dump_bfloat((__bf16)1.001953125f); // expected-warning{{Unknown}}
}
