// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix -Wno-pointer-to-int-cast -verify %s
// expected-no-diagnostics

// Testing core functionality of the SValBuilder.

int SValBuilderLogicNoCrash(int *x) {
  return 3 - (int)(x +3);
}

// http://llvm.org/bugs/show_bug.cgi?id=15863
// Don't crash when mixing 'bool' and 'int' in implicit comparisons to 0.
void pr15863(void) {
  extern int getBool(void);
  _Bool a = getBool();
  (void)!a; // no-warning
}
