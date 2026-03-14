// RUN: %check_clang_tidy -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c
// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++

// RUN: %check_clang_tidy -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c -fsigned-char
// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++ -fsigned-char

// RUN: %check_clang_tidy -std=c99 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c -funsigned-char
// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64 c++ -funsigned-char

long t0(char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t1(char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t2(unsigned char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t3(unsigned char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t4(char a, unsigned char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t5(char a, unsigned char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t6(unsigned char a, unsigned char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t7(unsigned char a, unsigned char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t8(signed char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t9(signed char a, char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t10(char a, signed char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t11(char a, signed char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t12(signed char a, signed char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t13(signed char a, signed char b) {
  return a * b;
  // CHECK-NOTES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
