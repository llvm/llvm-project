// RUN: %check_clang_tidy -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++

_ExtInt(64) t0(_ExtInt(32) a, _ExtInt(32) b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type '_ExtInt(64)' of a multiplication performed in type '_ExtInt(32)'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned _ExtInt(64) t1(_ExtInt(32) a, _ExtInt(32) b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned _ExtInt(64)' of a multiplication performed in type '_ExtInt(32)'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
_ExtInt(64) t2(unsigned _ExtInt(32) a, unsigned _ExtInt(32) b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10:  warning: performing an implicit widening conversion to type '_ExtInt(64)' of a multiplication performed in type 'unsigned _ExtInt(32)'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
