// RUN: %check_clang_tidy -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++

long t0(short a, int b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
long t1(short a, short b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
