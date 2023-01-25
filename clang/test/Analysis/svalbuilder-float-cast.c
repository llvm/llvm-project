// RUN: %clang_analyze_cc1 -analyzer-checker debug.ExprInspection -Wno-deprecated-non-prototype -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker debug.ExprInspection -Wno-deprecated-non-prototype -verify %s \
// RUN:    -analyzer-config support-symbolic-integer-casts=true

void clang_analyzer_denote(int, const char *);
void clang_analyzer_express(int);
void clang_analyzer_dump(int);
void clang_analyzer_dump_ptr(int *);

void SymbolCast_of_float_type_aux(int *p) {
  clang_analyzer_dump_ptr(p); // expected-warning {{&x}}
  clang_analyzer_dump(*p); // expected-warning {{Unknown}}
  // Storing to the memory region of 'float x' as 'int' will
  // materialize a fresh conjured symbol to regain accuracy.
  *p += 0;
  clang_analyzer_dump_ptr(p); // expected-warning {{&x}}
  clang_analyzer_dump(*p); // expected-warning {{conj_$0{int}}
  clang_analyzer_denote(*p, "$x");

  *p += 1;
  // This should NOT be (float)$x + 1. Symbol $x was never casted to float.
  clang_analyzer_express(*p); // expected-warning{{$x + 1}}
}

void SymbolCast_of_float_type(void) {
  extern float x;
  void (*f)() = SymbolCast_of_float_type_aux;
  f(&x);
}
