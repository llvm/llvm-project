// RUN: %clang_cc1 -triple x86_64-linux -O1 -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -new-struct-path-tbaa \
// RUN:   -relaxed-aliasing -emit-llvm %s -o - | FileCheck %s --check-prefix=OLD
// RUN: %clang_cc1 -triple x86_64-linux -O1 -new-struct-path-tbaa \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NEW
// RUN: %clang_cc1 -triple x86_64-linux -O1 -disable-llvm-passes \
// RUN:   -new-struct-path-tbaa -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IRGEN

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
// IRGEN-LABEL: define{{.*}} i32 @different_storage(
// IRGEN: %bf.load = load i8, ptr %a1, align 4, !tbaa [[C_A1:![0-9]+]]
// IRGEN: store i8 %bf.set, ptr %a1, align 4, !tbaa [[C_A1]]
// IRGEN: %bf.load4 = load i8, ptr %a2, align 4, !tbaa [[D_A2:![0-9]+]]
// IRGEN: store i8 %bf.set6, ptr %a2, align 4, !tbaa [[D_A2]]
// IRGEN: %bf.load11 = load i8, ptr %a110, align 4, !tbaa [[C_A1]]
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
// IRGEN-LABEL: define{{.*}} i32 @same_storage(
// IRGEN: %bf.load = load i8, ptr %a1, align 4, !tbaa [[C_A1]]
// IRGEN: store i8 %bf.set, ptr %a1, align 4, !tbaa [[C_A1]]
// IRGEN: %bf.load5 = load i8, ptr %a14, align 4, !tbaa [[D_A1:![0-9]+]]
// IRGEN: store i8 %bf.set7, ptr %a14, align 4, !tbaa [[D_A1]]
// IRGEN: %bf.load12 = load i8, ptr %a111, align 4, !tbaa [[C_A1]]
int same_storage(int i, int j) {
  c->b[i].a1.a = 1;
  d->c.b[j].a1.a = 0;
  return c->b[i].a1.a;
}

struct Bits {
  int a : 3;
  int b : 3;
};

// An access through int * cannot alias the enclosing struct object. Preserve
// that fact for the bit-field access, so the reload can be eliminated.
// OLD-LABEL: define{{.*}} i32 @scalar_cannot_alias_bitfield(
// OLD: load i8, ptr
// OLD: ret i32
// NEW-LABEL: define{{.*}} i32 @scalar_cannot_alias_bitfield(
// NEW: store i8
// NEW: store i32 1, ptr %y
// NEW-NOT: load i8, ptr
// NEW: ret i32 0
// IRGEN-LABEL: define{{.*}} i32 @scalar_cannot_alias_bitfield(
// IRGEN: store i8 %bf.set, ptr {{%.*}}, align 4, !tbaa [[BITS_A:![0-9]+]]
// IRGEN: store i32 1, ptr {{%.*}}, align 4, !tbaa [[INT:![0-9]+]]
// IRGEN: %bf.load1 = load i8, ptr {{%.*}}, align 4, !tbaa [[BITS_A]]
int scalar_cannot_alias_bitfield(struct Bits *x, int *y) {
  x->a = 0;
  *y = 1;
  return x->a;
}

// IRGEN-DAG: [[C_A1]] = !{[[C:![0-9]+]], [[CHAR:![0-9]+]], i64 0, i64 1}
// IRGEN-DAG: [[D_A2]] = !{[[D:![0-9]+]], [[CHAR]], i64 4, i64 1}
// IRGEN-DAG: [[D_A1]] = !{[[D]], [[CHAR]], i64 0, i64 1}
// IRGEN-DAG: [[BITS_A]] = !{[[BITS:![0-9]+]], [[CHAR]], i64 0, i64 1}
// IRGEN-DAG: [[INT]] = !{[[INT_TYPE:![0-9]+]], [[INT_TYPE]], i64 0, i64 4}
// IRGEN-DAG: [[A:![0-9]+]] = !{[[CHAR]], i64 4, !"A", [[CHAR]], i64 0, i64 1}
