// RUN: %check_clang_tidy %s google-runtime-float %t

long double foo;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'long double' type is not portable and should not be used [google-runtime-float]

typedef long double MyLongDouble;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: 'long double' type is not portable and should not be used [google-runtime-float]

typedef long double MyOtherLongDouble; // NOLINT

template <typename T>
void tmpl() { T i; }

long volatile double v = 10;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'volatile long double' type is not portable and should not be used [google-runtime-float]

long double h(long const double aaa, long double bbb = 0.5L) {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: 'long double' type is not portable and should not be used [google-runtime-float]
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: 'const long double' type is not portable and should not be used [google-runtime-float]
  // CHECK-MESSAGES: :[[@LINE-3]]:38: warning: 'long double' type is not portable and should not be used [google-runtime-float]
  // CHECK-MESSAGES: :[[@LINE-4]]:56: warning: 'long double' type from literal suffix 'L' is not portable and should not be used [google-runtime-float]
  double x = 0.1;
  double y = 0.2L;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'long double' type from literal suffix 'L' is not portable and should not be used [google-runtime-float]
#define ldtype long double
  ldtype z;
  tmpl<long double>();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'long double' type is not portable and should not be used [google-runtime-float]
  return 0;
}

struct S{};
constexpr S operator"" _baz(unsigned long long) {
  long double j;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'long double' type is not portable and should not be used [google-runtime-float]
  MyOtherLongDouble x;
  long int a = 1L;
  return S{};
}

