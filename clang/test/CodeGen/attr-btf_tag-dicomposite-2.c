// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64 -g -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_decl_tag("tag1")))
#define __tag2 __attribute__((btf_decl_tag("tag2")))

struct __tag1 __tag2 t1;

int foo(struct t1 *arg) {
  return (int)(long)arg;
}

// CHECK: define dso_local i32 @foo(
// CHECK-NOT: annotations
