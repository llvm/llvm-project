// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify -x c -std=c2y %s
// expected-no-diagnostics

// Ensure that VLA types are correctly handled by unary type traits in the
// expression engine. Previously, __datasizeof and _Countof both caused failed
// assertions.
void gh151711(int i) {
  (void)sizeof(int[i++]);

#ifdef __cplusplus
  // __datasizeof is only available in C++.
  (void)__datasizeof(int[i++]);
#else
  // _Countof is only available in C.
  (void)_Countof(int[i++]);
#endif
}
