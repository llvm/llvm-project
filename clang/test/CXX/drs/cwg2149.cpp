// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s --check-prefixes CXX98
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace cwg2149 { // cwg2149: 3.1
#if __cplusplus <= 201103L
struct X { int i, j, k; };
#else
struct X { int i, j, k = 42; };
#endif

template<int N> 
void f1(const X(&)[N]); // #cwg2149-f1

template<int N>
void f2(const X(&)[N][2]); // #cwg2149-f2

void f() {
  X a[] = { 1, 2, 3, 4, 5, 6 };
  static_assert(sizeof(a) / sizeof(X) == 2, "");
  X b[2] = { { 1, 2, 3 }, { 4, 5, 6 } };
  X c[][2] = { 1, 2, 3, 4, 5, 6 };
  static_assert(sizeof(c) / sizeof(X[2]) == 1, "");
  
  #if __cplusplus >= 201103L
  constexpr X ca[] = { 1, 2, 3, 4, 5, 6 };
  constexpr X cb[2] = { { 1, 2, 3 }, { 4, 5, 6 } };
  static_assert(ca[0].i == cb[0].i, "");
  static_assert(ca[0].j == cb[0].j, "");
  static_assert(ca[0].k == cb[0].k, "");
  static_assert(ca[1].i == cb[1].i, "");
  static_assert(ca[1].j == cb[1].j, "");
  static_assert(ca[1].k == cb[1].k, "");

  f1({ 1, 2, 3, 4, 5, 6 });
  // since-cxx11-error@-1 {{no matching function for call to 'f1'}}
  //   since-cxx11-note@#cwg2149-f1 {{candidate function [with N = 6] not viable: no known conversion from 'int' to 'const X' for 1st argument}}
  f2({ 1, 2, 3, 4, 5, 6 });
  // since-cxx11-error@-1 {{no matching function for call to 'f2'}}
  //   since-cxx11-note@#cwg2149-f2 {{candidate function [with N = 6] not viable: no known conversion from 'int' to 'const X[2]' for 1st argument}}
  #endif
}
} // namespace cwg2149

// Constant evaluation is not powerful enough in 98 mode to check for equality
// via static_assert, even with constant folding enabled.

// CXX98:       VarDecl {{.+}} a 'X[2]'
// CXX98-NEXT:  `-InitListExpr {{.+}} 'X[2]'
// CXX98-NEXT:    |-InitListExpr {{.+}} 'X':'cwg2149::X'
// CXX98-NEXT:    | |-IntegerLiteral {{.+}} 'int' 1
// CXX98-NEXT:    | |-IntegerLiteral {{.+}} 'int' 2
// CXX98-NEXT:    | `-IntegerLiteral {{.+}} 'int' 3
// CXX98-NEXT:    `-InitListExpr {{.+}} 'X':'cwg2149::X'
// CXX98-NEXT:      |-IntegerLiteral {{.+}} 'int' 4
// CXX98-NEXT:      |-IntegerLiteral {{.+}} 'int' 5
// CXX98-NEXT:      `-IntegerLiteral {{.+}} 'int' 6

// CXX98:       VarDecl {{.+}} b 'X[2]'
// CXX98-NEXT:  `-InitListExpr {{.+}} 'X[2]'
// CXX98-NEXT:    |-InitListExpr {{.+}} 'X':'cwg2149::X'
// CXX98-NEXT:    | |-IntegerLiteral {{.+}} 'int' 1
// CXX98-NEXT:    | |-IntegerLiteral {{.+}} 'int' 2
// CXX98-NEXT:    | `-IntegerLiteral {{.+}} 'int' 3
// CXX98-NEXT:    `-InitListExpr {{.+}} 'X':'cwg2149::X'
// CXX98-NEXT:      |-IntegerLiteral {{.+}} 'int' 4
// CXX98-NEXT:      |-IntegerLiteral {{.+}} 'int' 5
// CXX98-NEXT:      `-IntegerLiteral {{.+}} 'int' 6
