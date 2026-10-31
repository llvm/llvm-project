// RUN: %check_clang_tidy -check-suffixes=ALL,C -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy -check-suffixes=ALL,CXX %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++

long t0(short a, int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
long t1(short a, short b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

// Both operands are unsigned, and the multiplication only became a signed
// 'int' due to integer promotion; the suggested wider type should stay
// unsigned instead of switching signedness domains.
unsigned long t2(unsigned short a, unsigned short b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (unsigned long)( )
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (unsigned long)
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
}

long t3(unsigned short a, unsigned short b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (long)( )
  // CHECK-NOTES-CXX:                  static_cast<long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (unsigned long)
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
}

