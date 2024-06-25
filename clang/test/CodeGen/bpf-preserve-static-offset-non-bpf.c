// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Verify that __attribute__((preserve_static_offset))
// has no effect for non-BPF target.

#define __ctx __attribute__((preserve_static_offset))

struct foo {
  int a;
} __ctx;

// CHECK-NOT: @llvm_preserve_static_offset

int bar(struct foo *p) {
  return p->a;
}
