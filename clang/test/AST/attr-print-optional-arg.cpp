// RUN: %clang_cc1 -std=c++20 -fdeclspec %s -ast-print | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -fdeclspec -emit-pch -o %t.ast %s
// RUN: %clang_cc1 -std=c++20 -fdeclspec %t.ast -ast-print | FileCheck %s

// Check that optional attribute arguments which were not written are omitted
// when printing, instead of being printed with an empty/default value.

// CHECK: {{\[\[}}nodiscard]] int f1();
[[nodiscard]] int f1();

// CHECK: {{\[\[}}nodiscard("no")]] int f2();
[[nodiscard("no")]] int f2();

// CHECK: {{\[\[}}nodiscard]] int f3();
[[nodiscard("")]] int f3();

// CHECK: __attribute__((unavailable)) int f4();
__attribute__((unavailable)) int f4();

// CHECK: __attribute__((unavailable("gone"))) int f5();
__attribute__((unavailable("gone"))) int f5();

// CHECK: {{\[\[}}deprecated]] int f6();
[[deprecated]] int f6();

// CHECK: {{\[\[}}deprecated("msg")]] int f7();
[[deprecated("msg")]] int f7();

// An explicitly written empty message is not distinguishable from an omitted
// one, so it is omitted as well.
// CHECK: {{\[\[}}deprecated]] int f8();
[[deprecated("")]] int f8();

// CHECK: __attribute__((deprecated)) int f9();
__attribute__((deprecated)) int f9();

// CHECK: __attribute__((deprecated)) int f10();
__attribute__((deprecated(""))) int f10();

// CHECK: __attribute__((deprecated("msg"))) int f11();
__attribute__((deprecated("msg"))) int f11();

// The replacement is positional, so an empty message must be kept if there is
// a replacement.
// CHECK: __attribute__((deprecated("", "repl"))) int f12();
__attribute__((deprecated("", "repl"))) int f12();

// CHECK: {{\[\[}}gnu::deprecated]] int f13();
[[gnu::deprecated]] int f13();

// CHECK: {{\[\[}}gnu::deprecated("msg")]] int f14();
[[gnu::deprecated("msg")]] int f14();

// CHECK: __declspec(deprecated) int f15();
__declspec(deprecated) int f15();

// CHECK: __declspec(deprecated("msg")) int f16();
__declspec(deprecated("msg")) int f16();

// A non-optional string argument is always printed, even if it is empty.
// CHECK: __attribute__((annotate(""))) int f17();
__attribute__((annotate(""))) int f17();

// CHECK: struct {{\[\[}}nodiscard]] S {
// CHECK-NEXT: };
struct [[nodiscard]] S {};
