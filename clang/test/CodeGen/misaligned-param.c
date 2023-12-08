// RUN: %clang_cc1 %s -std=c89 -triple i386-apple-darwin -emit-llvm -o - | FileCheck %s
// Misaligned parameter must be memcpy'd to correctly aligned temporary.

struct s { int x; long double y; };
int bar(struct s *, struct s *);
long double foo(struct s x, int i, struct s y) {
// CHECK: foo
// CHECK: %x = alloca %struct.s, align 16
// CHECK: %y = alloca %struct.s, align 16
// CHECK: memcpy
// CHECK: memcpy
// CHECK: bar
  return bar(&x, &y);
}
