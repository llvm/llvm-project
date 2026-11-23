// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-implicit-widening-of-multiplication-result.UseCXXHeadersInCppSources: false, \
// RUN:     bugprone-implicit-widening-of-multiplication-result.IncludeStyle: google \
// RUN:   }}" -- -target x86_64-unknown-unknown -x c++

long mul(int a, int b) {
  return a * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int' [bugprone-implicit-widening-of-multiplication-result]
  // CHECK-MESSAGES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-MESSAGES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long ptr_off(int a, int b, char *p) {
  return p[a * b];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ptrdiff_t' [bugprone-implicit-widening-of-multiplication-result]
  // CHECK-MESSAGES: :[[@LINE-2]]:12: note: make conversion explicit to silence this warning
  // CHECK-MESSAGES: :[[@LINE-3]]:12: note: perform multiplication in a wider type
}
