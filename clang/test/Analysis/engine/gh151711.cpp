// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -x c %s

void clang_analyzer_dump(int);

// Ensure that VLA types are correctly handled by unary type traits in the
// expression engine. Previously, __datasizeof and _Countof both caused failed
// assertions.
void gh151711(int i) {
  clang_analyzer_dump(sizeof(int[i++]));       // expected-warning {{Unknown}}
#ifdef __cplusplus
  // __datasizeof is only available in C++.
  clang_analyzer_dump(__datasizeof(int[i++])); // expected-warning {{Unknown}}
#else
  // _Countof is only available in C.
  clang_analyzer_dump(_Countof(int[i++]));     // expected-warning {{Unknown}}
#endif
}
