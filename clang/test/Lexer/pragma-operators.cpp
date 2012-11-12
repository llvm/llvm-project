// RUN: %clang_cc1 -fms-extensions -E %s | FileCheck %s

// Test that we properly expand the C99 _Pragma and Microsoft __pragma
// into #pragma directives, with newlines where needed. <rdar://problem/8412013>

// CHECK: #line
// CHECK: #pragma warning(push)
// CHECK: extern "C" {
// CHECK: #line
// CHECK: #pragma warning(push)
// CHECK:  int foo() { return 0; } }
// CHECK: #pragma warning(pop)
#define A(X) extern "C" { __pragma(warning(push)) \
  int X() { return 0; } \
}
#define B(X) A(X)
#pragma warning(push)
B(foo)
#pragma warning(pop)
