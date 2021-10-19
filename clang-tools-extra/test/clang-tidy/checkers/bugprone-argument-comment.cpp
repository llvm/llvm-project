// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- -- -I %S/Inputs/bugprone-argument-comment

// FIXME: clang-tidy should provide a -verify mode to make writing these checks
// easier and more accurate.

void ffff(int xxxx, int yyyy);

void f(int x, int y);
void g() {
  // CHECK-NOTES: [[@LINE+4]]:5: warning: argument name 'y' in comment does not match parameter name 'x'
  // CHECK-NOTES: [[@LINE-3]]:12: note: 'x' declared here
  // CHECK-NOTES: [[@LINE+2]]:14: warning: argument name 'z' in comment does not match parameter name 'y'
  // CHECK-NOTES: [[@LINE-5]]:19: note: 'y' declared here
  f(/*y=*/0, /*z=*/0);
  // CHECK-FIXES: {{^}}  f(/*y=*/0, /*z=*/0);

  f(/*x=*/1, /*y=*/1);

  ffff(0 /*aaaa=*/, /*bbbb*/ 0); // Unsupported formats.
}

struct C {
  C(int x, int y);
};
C c(/*x=*/0, /*y=*/0);

struct Closure {};

template <typename T1, typename T2>
Closure *NewCallback(void (*f)(T1, T2), T1 arg1, T2 arg2) { return nullptr; }

template <typename T1, typename T2>
Closure *NewPermanentCallback(void (*f)(T1, T2), T1 arg1, T2 arg2) { return nullptr; }

void h() {
  (void)NewCallback(&ffff, /*xxxx=*/11, /*yyyy=*/22);
  (void)NewPermanentCallback(&ffff, /*xxxx=*/11, /*yyyy=*/22);
}

template<typename... Args>
void variadic(Args&&... args);

template<typename... Args>
void variadic2(int zzz, Args&&... args);

void templates() {
  variadic(/*xxx=*/0, /*yyy=*/1);
  variadic2(/*zzU=*/0, /*xxx=*/1, /*yyy=*/2);
  // CHECK-NOTES: [[@LINE-1]]:13: warning: argument name 'zzU' in comment does not match parameter name 'zzz'
  // CHECK-NOTES: :[[@LINE-6]]:20: note: 'zzz' declared here
  // CHECK-FIXES: variadic2(/*zzz=*/0, /*xxx=*/1, /*yyy=*/2);
}

#define FALSE 0
void qqq(bool aaa);
void f2() { qqq(/*bbb=*/FALSE); }
// CHECK-NOTES: [[@LINE-1]]:17: warning: argument name 'bbb' in comment does not match parameter name 'aaa'
// CHECK-NOTES: [[@LINE-3]]:15: note: 'aaa' declared here
// CHECK-FIXES: void f2() { qqq(/*bbb=*/FALSE); }

void f3(bool _with_underscores_);
void ignores_underscores() {
  f3(/*With_Underscores=*/false);
}

namespace IgnoresImplicit {
struct S {
  S(int x);
  int x;
};

struct T {
  // Use two arguments (one defaulted) because simplistic check for implicit
  // constructor looks for only one argument. We need to default the argument so
  // that it will still be triggered implicitly.  This is not contrived -- it
  // comes up in real code, for example std::set(std::initializer_list...).
  T(S s, int y = 0);
};

void k(T arg1);

void mynewtest() {
  int foo = 3;
  k(/*arg1=*/S(foo));
}
} // namespace IgnoresImplicit

namespace ThisEditDistanceAboveThreshold {
void f4(int xxx);
void g() { f4(/*xyz=*/0); }
// CHECK-NOTES: [[@LINE-1]]:15: warning: argument name 'xyz' in comment does not match parameter name 'xxx'
// CHECK-NOTES: [[@LINE-3]]:13: note: 'xxx' declared here
// CHECK-FIXES: void g() { f4(/*xyz=*/0); }
}

namespace OtherEditDistanceAboveThreshold {
void f5(int xxx, int yyy);
void g() { f5(/*Zxx=*/0, 0); }
// CHECK-NOTES: [[@LINE-1]]:15: warning: argument name 'Zxx' in comment does not match parameter name 'xxx'
// CHECK-NOTES: [[@LINE-3]]:13: note: 'xxx' declared here
// CHECK-FIXES: void g() { f5(/*xxx=*/0, 0); }
struct C2 {
  C2(int xxx, int yyy);
};
C2 c2(/*Zxx=*/0, 0);
// CHECK-NOTES: [[@LINE-1]]:7: warning: argument name 'Zxx' in comment does not match parameter name 'xxx'
// CHECK-NOTES: [[@LINE-4]]:10: note: 'xxx' declared here
// CHECK-FIXES: C2 c2(/*xxx=*/0, 0);
}

namespace OtherEditDistanceBelowThreshold {
void f6(int xxx, int yyy);
void g() { f6(/*xxy=*/0, 0); }
// CHECK-NOTES: [[@LINE-1]]:15: warning: argument name 'xxy' in comment does not match parameter name 'xxx'
// CHECK-NOTES: [[@LINE-3]]:13: note: 'xxx' declared here
// CHECK-FIXES: void g() { f6(/*xxy=*/0, 0); }
}


namespace std {
template <typename T>
class vector {
public:
  void assign(int __n, const T &__val);
};
template<typename T>
void swap(T& __a, T& __b);
} // namespace std
namespace ignore_std_functions {
void test(int a, int b) {
  std::vector<int> s;
  // verify the check is not fired on std functions.
  s.assign(1, /*value=*/2);
  std::swap(a, /*num=*/b);
}
} // namespace ignore_std_functions

namespace regular_header {
#include "header-with-decl.h"
void test() {
  my_header_function(/*not_arg=*/1);
// CHECK-NOTES: [[@LINE-1]]:22: warning: argument name 'not_arg' in comment does not match parameter name 'arg'
// CHECK-NOTES: header-with-decl.h:1:29: note: 'arg' declared here
// CHECK-FIXES: my_header_function(/*not_arg=*/1);
}
} // namespace regular_header

namespace system_header {
#include "system-header-with-decl.h"
void test() {
  my_system_header_function(/*not_arg=*/1);
}
} // namespace system_header

void testInvalidSlocCxxConstructExpr() {
  __builtin_va_list __args;
  // __builtin_va_list has no defination in any source file
} 
