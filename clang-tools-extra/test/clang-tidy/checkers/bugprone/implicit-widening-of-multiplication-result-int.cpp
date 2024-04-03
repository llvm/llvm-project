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

long t0(int a, int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (long)( )
  // CHECK-NOTES-CXX:                  static_cast<long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (long)
  // CHECK-NOTES-CXX:                  static_cast<long>( )
}
unsigned long t1(int a, int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (unsigned long)( )
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (long)
  // CHECK-NOTES-CXX:                  static_cast<long>( )
}

long t2(unsigned int a, int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (long)( )
  // CHECK-NOTES-CXX:                  static_cast<long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (unsigned long)
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
}
unsigned long t3(unsigned int a, int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-C:                    (unsigned long)( )
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
  // CHECK-NOTES-ALL: :[[@LINE-5]]:10: note: perform multiplication in a wider type
  // CHECK-NOTES-C:                    (unsigned long)
  // CHECK-NOTES-CXX:                  static_cast<unsigned long>( )
}

long t4(int a, unsigned int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t5(int a, unsigned int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t6(unsigned int a, unsigned int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
unsigned long t7(unsigned int a, unsigned int b) {
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'unsigned int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}

long t8(int a, int b) {
  return (a * b);
  // CHECK-NOTES-ALL: :[[@LINE-1]]:11: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:11: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:11: note: perform multiplication in a wider type
}
long t9(int a, int b) {
  return (a)*b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL: :[[@LINE-3]]:10: note: perform multiplication in a wider type
}
long n10(int a, int b) {
  return (long)(a * b);
}
long n11(int a, int b) {
  return (unsigned long)(a * b);
}

long n12(long a, int b) {
  return a * b;
}
long n13(int a, long b) {
  return a * b;
}

long n14(int a, int b, int c) {
  return a + b * c;
}
long n15(int a, int b, int c) {
  return a * b + c;
}

#ifdef __cplusplus
template <typename T1, typename T2>
T2 template_test(T1 a, T1 b) {
  return a * b;
}
long template_test_instantiation(int a, int b) {
  return template_test<int, long>(a, b);
}
#endif
