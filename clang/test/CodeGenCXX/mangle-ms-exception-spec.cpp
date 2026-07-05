// RUN: %clang_cc1 -std=c++11 -fms-extensions -emit-llvm %s -o - -triple=x86_64-pc-win32 -Wno-noexcept-type -fms-compatibility-version=19.12 | FileCheck %s --check-prefix=CHECK --check-prefix=CXX11
// RUN: %clang_cc1 -std=c++17 -fms-extensions -emit-llvm %s -o - -triple=x86_64-pc-win32 | FileCheck %s --check-prefix=CHECK --check-prefix=NOCOMPAT
// RUN: %clang_cc1 -std=c++17 -fms-extensions -emit-llvm %s -o - -triple=x86_64-pc-win32 -fms-compatibility-version=19.12 | FileCheck %s --check-prefix=CHECK --check-prefix=CXX17

// Prove that mangling only changed for noexcept types under /std:C++17, not all noexcept functions
// CHECK-DAG: @"?nochange@@YAXXZ"
void nochange() noexcept {}

// CXX11-DAG: @"?a@@YAXP6AHXZ@Z"
// NOCOMPAT-DAG: @"?a@@YAXP6AHXZ@Z"
// CXX17-DAG: @"?a@@YAXP6AHX_E@Z"
void a(int() noexcept) {}
// CHECK-DAG: @"?b@@YAXP6AHXZ@Z"
void b(int() noexcept(false)) {}
// CXX11-DAG: @"?c@@YAXP6AHXZ@Z"
// NOCOMPAT-DAG: @"?c@@YAXP6AHXZ@Z"
// CXX17-DAG: @"?c@@YAXP6AHX_E@Z"
void c(int() noexcept(true)) {}
// CHECK-DAG: @"?d@@YAXP6AHXZ@Z"
void d(int()) {}

template <typename T>
class e;
template <typename T, typename... U>
class e<T(U...) noexcept> {
  // CXX11-DAG: @"?ee@?$e@$$A6AXXZ@@EEAAXXZ"
  // NOCOMPAT-DAG: @"?ee@?$e@$$A6AXXZ@@EEAAXXZ"
  // CXX17-DAG: @"?ee@?$e@$$A6AXX_E@@EEAAXXZ"
  virtual T ee(U &&...) noexcept {};
};

e<void() noexcept> e1;

template <typename T>
class f;
template <typename T, typename... U>
class f<T(U...)> {
  // CHECK-DAG: @"?ff@?$f@$$A6AXXZ@@EEAAXXZ"
  virtual T ff(U &&...) noexcept {};
};

f<void()> f1;
