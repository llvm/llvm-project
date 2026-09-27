// RUN: %clang_cc1 -triple x86_64-linux -O1 -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -new-struct-path-tbaa \
// RUN:   -relaxed-aliasing -emit-llvm %s -o - | FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -new-struct-path-tbaa \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NEW

struct A {
  int a : 3;
  int b : 3;
};
struct B {
  struct A a1, a2;
};
struct C {
  struct B b[10];
} *c;
struct D {
  struct C c;
} *d;

// The two bit-fields occupy different storage units. Their TBAA tags retain
// enough of the enclosing struct path to prove that the store cannot clobber
// the value written by the first store.
// OLD-LABEL: define{{.*}} i32 @different_storage(
// OLD: load i8, ptr
// OLD: ret i32
// NEW-LABEL: define{{.*}} i32 @different_storage(
// NEW-COUNT-2: load i8, ptr
// NEW: ret i32 0
int different_storage(int i, int j) {
  c->b[i].a1.a = 0;
  d->c.b[j].a2.b = 1;
  return c->b[i].a1.a;
}

// These accesses may designate the same storage unit, so the reload must be
// retained.
// NEW-LABEL: define{{.*}} i32 @same_storage(
// NEW: store i8
// NEW: load i8, ptr
// NEW: ret i32
int same_storage(int i, int j) {
  c->b[i].a1.a = 1;
  d->c.b[j].a1.a = 0;
  return c->b[i].a1.a;
}
