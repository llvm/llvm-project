// RUN: %clang_cc1 -triple armv7-apple-darwin -emit-llvm %s -o - | FileCheck %s

// CHECK: call i32 asm "foo0", {{.*}} [[READNONE:#[0-9]+]]
// CHECK: call i32 asm "foo1", {{.*}} [[READNONE]]
// CHECK: call i32 asm "foo2", {{.*}} [[NOATTRS:#[0-9]+]]
// CHECK: call i32 asm sideeffect "foo3", {{.*}} [[INACCESSIBLEMEMONLY:#[0-9]+]]
// CHECK: call i32 asm "foo4", {{.*}} [[ARGREAD:#[0-9]+]]
// CHECK: call i32 asm "foo5", {{.*}} [[ARGREAD]]
// CHECK: call i32 asm "foo6", {{.*}} [[ARGWRITE:#[0-9]+]]
// CHECK: call void asm sideeffect "foo7", {{.*}} [[INACCESSIBLEMEMONLY]]
// CHECK: call i32 asm "foo8", {{.*}} [[READNONE]]
// CHECK: call void asm "foo9", {{.*}} [[ARGREADWRITE:#[0-9]+]]
// CHECK: call void asm "foo10", {{.*}} [[ARGREADWRITE]]

// CHECK: attributes [[READNONE]] = { nounwind willreturn memory(none) }
// CHECK: attributes [[NOATTRS]] = { nounwind willreturn }
// CHECK: attributes [[INACCESSIBLEMEMONLY]] = { nounwind memory(inaccessiblemem: readwrite) }
// CHECK: attributes [[ARGREAD]] = { nounwind willreturn memory(argmem: read) }
// CHECK: attributes [[ARGWRITE]] = { nounwind willreturn memory(argmem: write) }
// CHECK: attributes [[ARGREADWRITE]] = { nounwind willreturn memory(argmem: readwrite) }

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
  __asm__ ("foo9" : "=m"(g1) : "m"(g0));
  __asm__ ("foo10" : "+m"(g1));
}
