// RUN: %clang_analyze_cc1 -analyzer-checker debug.ExprInspection -verify %s
void clang_analyzer_denote(int, const char *);
void clang_analyzer_express(int);

void SymbolCast_of_float_type_aux(int *p) {
  *p += 0;
  // FIXME: Ideally, all unknown values should be symbolicated.
  clang_analyzer_denote(*p, "$x"); // expected-warning{{Not a symbol}}

  *p += 1;
  // This should NOT be (float)$x + 1. Symbol $x was never casted to float.
  // FIXME: Ideally, this should be $x + 1.
  clang_analyzer_express(*p); // expected-warning{{Not a symbol}}
}

void SymbolCast_of_float_type(void) {
  extern float x;
  void (*f)() = SymbolCast_of_float_type_aux;
  f(&x);
}
