// RUN: %check_clang_tidy -check-suffixes=ALL,C -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy -check-suffixes=ALL,CXX %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++

// RUN: %check_clang_tidy -check-suffixes=ALL,C -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         bugprone-implicit-widening-of-multiplication-result.UseCXXStaticCastsInCppSources: false \
// RUN:     }}' -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy -check-suffixes=ALL,C %s bugprone-implicit-widening-of-multiplication-result %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         bugprone-implicit-widening-of-multiplication-result.UseCXXStaticCastsInCppSources: false \
// RUN:     }}' -- -target x86_64-unknown-unknown -x c++

char *t0(char *base, int a, int b) {
  return base + a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:17: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (ptrdiff_t)( )
  // CHECK-NOTES-CXX:                  static_cast<ptrdiff_t>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:17: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (ptrdiff_t)
  // CHECK-NOTES-CXX:                  static_cast<ptrdiff_t>( )
}
char *t1(char *base, int a, int b) {
  return a * b + base;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

char *t2(char *base, unsigned int a, int b) {
  return base + a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'unsigned int' is used as a pointer offset after an implicit widening conversion to type 'size_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:17: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (size_t)( )
  // CHECK-NOTES-CXX:                  static_cast<size_t>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:17: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (size_t)
  // CHECK-NOTES-CXX:                  static_cast<size_t>( )
}

char *t3(char *base, int a, unsigned int b) {
  return base + a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'unsigned int' is used as a pointer offset after an implicit widening conversion to type 'size_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:17: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:17: note: perform multiplication in a wider type
}

char *t4(char *base, unsigned int a, unsigned int b) {
  return base + a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'unsigned int' is used as a pointer offset after an implicit widening conversion to type 'size_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:17: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:17: note: perform multiplication in a wider type
}

char *t5(char *base, int a, int b, int c) {
  return base + a * b + c;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:17: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:17: note: perform multiplication in a wider type
}
char *t6(char *base, int a, int b, int c) {
  return base + a + b * c;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:21: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:21: note: perform multiplication in a wider type
}

char *n7(char *base, int a, int b) {
  return base + (a * b);
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:18: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:18: note: perform multiplication in a wider type
}
char *n8(char *base, int a, int b, int c) {
  return base + (a * b) + c;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:18: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:18: note: perform multiplication in a wider type
}
char *n9(char *base, int a, int b, int c) {
  return base + (a * b + c);
}

char *n10(char *base, int a, int b) {
  return base + (long)(a * b);
}
char *n11(char *base, int a, int b) {
  return base + (unsigned long)(a * b);
}

#ifdef __cplusplus
template <typename T>
char *template_test(char *base, T a, T b) {
  return base + a * b;
}
char *template_test_instantiation(char *base, int a, int b) {
  return template_test(base, a, b);
}
#endif
