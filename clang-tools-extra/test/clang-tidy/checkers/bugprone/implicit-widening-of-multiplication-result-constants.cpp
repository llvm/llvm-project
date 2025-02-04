// RUN: %check_clang_tidy %s bugprone-implicit-widening-of-multiplication-result %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         bugprone-implicit-widening-of-multiplication-result.IgnoreConstantIntExpr: true \
// RUN:     }}' -- -target x86_64-unknown-unknown -x c++

long t0() {
  return 1 * 4;
}

unsigned long t1() {
  const int a = 2;
  const int b = 3;
  return a * b;
}

long t2() {
  constexpr int a = 16383; // ~1/2 of int16_t max
  constexpr int b = 2;
  return a * b;
}

constexpr int global_value() {
  return 16;
}

unsigned long t3() {
  constexpr int a = 3;
  return a * global_value();
}

long t4() {
  const char a = 3;
  const short b = 2;
  const int c = 5;
  return c * b * a;
}

long t5() {
  constexpr int min_int = (-2147483647 - 1); // A literal of -2147483648 evaluates to long
  return 1 * min_int;
}

unsigned long n0() {
  const int a = 1073741824; // 1/2 of int32_t max
  const int b = 3;
  return a * b;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'unsigned long' of a multiplication performed in type 'int'
  // CHECK-MESSAGES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-MESSAGES:                  static_cast<unsigned long>( )
  // CHECK-MESSAGES: :[[@LINE-4]]:10: note: perform multiplication in a wider type
}

double n1() {
  const long a = 100000000;
  return a * 400;
}
