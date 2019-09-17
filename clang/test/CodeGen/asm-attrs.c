// RUN: %clang_cc1 -triple armv7-apple-darwin -emit-llvm %s -o - | FileCheck %s

// CHECK: call i32 asm "foo0", {{.*}} [[READNONE:#[0-9]+]]
// CHECK: call i32 asm "foo1", {{.*}} [[READNONE]]
// CHECK: call i32 asm "foo2", {{.*}} [[NOATTRS:#[0-9]+]]
// CHECK: call i32 asm sideeffect "foo3", {{.*}} [[NOATTRS]]
// CHECK: call i32 asm "foo4", {{.*}} [[READONLY:#[0-9]+]]
// CHECK: call i32 asm "foo5", {{.*}} [[READONLY]]
// CHECK: call i32 asm "foo6", {{.*}} [[NOATTRS]]
// CHECK: call void asm sideeffect "foo7", {{.*}} [[NOATTRS]]
// CHECK: call i32 asm "foo8", {{.*}} [[READNONE]]

// CHECK: attributes [[READNONE]] = { nounwind readnone }
// CHECK: attributes [[NOATTRS]] = { nounwind }
// CHECK: attributes [[READONLY]] = { nounwind readonly }

int g0, g1;

struct S {
  int i;
} g2;

void test_attrs(int a) {
  __asm__ ("foo0" : "=r"(g1) : "r"(a));
  __asm__ ("foo1" : "=r"(g1) : "r"(a) : "cc");
  __asm__ ("foo2" : "=r"(g1) : "r"(a) : "memory");
  __asm__ volatile("foo3" : "=r"(g1) : "r"(a));
  __asm__ ("foo4" : "=r"(g1) : "r"(a), "m"(g0));
  __asm__ ("foo5" : "=r"(g1) : "r"(a), "Q"(g0));
  __asm__ ("foo6" : "=r"(g1), "=m"(g0) : "r"(a));
  __asm__ ("foo7" : : "r"(a));
  __asm__ ("foo8" : "=r"(g2) : "r"(a));
}
