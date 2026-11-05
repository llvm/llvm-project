// RUN: %clang_cc1 -std=c++20 %s -ast-print | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t.ast %s
// RUN: %clang_cc1 -std=c++20 %t.ast -ast-print | FileCheck %s

// Check that optional attribute arguments which were not written are omitted
// when printing, instead of being printed with an empty/default value.
 
// CHECK: {{\[\[}}nodiscard]] int a();
[[nodiscard]] int a();

// CHECK: {{\[\[}}nodiscard("no")]] int b();
[[nodiscard("no")]] int b();

// CHECK: {{\[\[}}nodiscard]] int c();
[[nodiscard("")]] int c();

// CHECK: __attribute__((unavailable)) int d();
__attribute__((unavailable)) int d();

// CHECK: __attribute__((unavailable("gone"))) int e();
__attribute__((unavailable("gone"))) int e();

// FIXME: [[deprecated]] should print without the empty message, but
// deprecated is special-cased in ClangAttrEmitter.cpp
// (writeDeprecatedAttrValue), which always prints the message.
// CHECK: {{\[\[}}deprecated("")]] int f();
[[deprecated]] int f();

// CHECK: struct {{\[\[}}nodiscard]] S {
// CHECK-NEXT: };
struct [[nodiscard]] S {};
