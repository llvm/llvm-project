// RUN: %check_clang_tidy -check-suffixes=ALL,NI %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c++
// RUN: %check_clang_tidy -check-suffixes=ALL %s bugprone-implicit-widening-of-multiplication-result %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         bugprone-implicit-widening-of-multiplication-result.IgnoreConstantIntExpr: true \
// RUN:     }}' -- -target x86_64-unknown-unknown -x c++

long t0() {
  return 1 * 4;
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

unsigned long t1() {
  const int a = 2;
  const int b = 3;
  return a * b;
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<unsigned long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

long t2() {
  constexpr int a = 16383; // ~1/2 of int16_t max
  constexpr int b = 2;
  return a * b;
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

constexpr int global_value() {
  return 16;
}

unsigned long t3() {
  constexpr int a = 3;
  return a * global_value();
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<unsigned long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

long t4() {
  const char a = 3;
  const short b = 2;
  const int c = 5;
  return c * b * a;
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

long t5() {
  constexpr int min_int = (-2147483647 - 1); // A literal of -2147483648 evaluates to long
  return 1 * min_int;
  // CHECK-NOTES-NI: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  // CHECK-NOTES-NI: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-NI:                  static_cast<long>( )
  // CHECK-NOTES-NI: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

unsigned long n0() {
  const int a = 1073741824; // 1/2 of int32_t max
  const int b = 3;
  return a * b;
  // CHECK-NOTES-ALL: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-NOTES-ALL: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-NOTES-ALL:                  static_cast<unsigned long>( )
  // CHECK-NOTES-ALL: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

double n1() {
  const long a = 100000000;
  return a * 400;
}
