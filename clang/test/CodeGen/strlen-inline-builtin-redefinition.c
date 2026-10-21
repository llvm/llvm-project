// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// C accepts a second `extern inline gnu_inline` definition of a builtin. Both
// definitions reach EmitGlobalFunctionDefinition when the builtin is used in
// between; only the first body is emitted into the `.inline` clone.

typedef unsigned long size_t;

extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
size_t strlen(const char *s) {
  return 1;
}

size_t call_strlen_1(const char *s) {
  return strlen(s);
}

extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
size_t strlen(const char *s) {
  return 2;
}

size_t call_strlen_2(const char *s) {
  return strlen(s);
}

// CHECK-LABEL: define {{.*}}i64 @call_strlen_1(
// CHECK: call i64 @strlen.inline(
//
// CHECK-LABEL: define internal i64 @strlen.inline(
// CHECK: ret i64 1
// CHECK-NOT: ret i64 2
//
// CHECK-LABEL: define {{.*}}i64 @call_strlen_2(
// CHECK: call i64 @strlen.inline(
