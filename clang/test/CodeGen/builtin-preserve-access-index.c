// RUN: %clang_cc1 -triple x86_64 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

#define _(x) (__builtin_preserve_access_index(x))

const void *unit1(const void *arg) {
  return _(arg);
}
// CHECK: define dso_local ptr @unit1
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit2(void) {
  return _((const void *)0xffffffffFFFF0000ULL);
}
// CHECK: define dso_local ptr @unit2
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit3(const int *arg) {
  return _(arg + 1);
}
// CHECK: define dso_local ptr @unit3
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit4(const int *arg) {
  return _(&arg[1]);
}
// CHECK: define dso_local ptr @unit4
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(i32) %{{[0-9a-z]+}}, i32 0, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[POINTER:[0-9]+]]

const void *unit5(const int *arg[5]) {
  return _(&arg[1][2]);
}
// CHECK: define dso_local ptr @unit5
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(ptr) %{{[0-9a-z]+}}, i32 0, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(i32) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[POINTER:[0-9]+]]

struct s1 {
  char a;
  int b;
};

struct s2 {
  char a1:1;
  char a2:1;
  int b;
};

struct s3 {
  char a1:1;
  char a2:1;
  char :6;
  int b;
};

const void *unit6(struct s1 *arg) {
  return _(&arg->a);
}
// CHECK: define dso_local ptr @unit6
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]

const void *unit7(struct s1 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local ptr @unit7
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1]]

const void *unit8(struct s2 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local ptr @unit8
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S2:[0-9]+]]

const void *unit9(struct s3 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local ptr @unit9
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s3) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S3:[0-9]+]]

union u1 {
  char a;
  int b;
};

union u2 {
  char a;
  int :32;
  int b;
};

const void *unit10(union u1 *arg) {
  return _(&arg->a);
}
// CHECK: define dso_local ptr @unit10
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1:[0-9]+]]

const void *unit11(union u1 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local ptr @unit11
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1]]

const void *unit12(union u2 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local ptr @unit12
// CHECK-NOT: getelementptr
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U2:[0-9]+]]

struct s4 {
  char d;
  union u {
    int b[4];
    char a;
  } c;
};

union u3 {
  struct s {
    int b[4];
  } c;
  char a;
};

const void *unit13(struct s4 *arg) {
  return _(&arg->c.b[2]);
}
// CHECK: define dso_local ptr @unit13
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S4:[0-9]+]]
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_I_U:[0-9]+]]
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}

const void *unit14(union u3 *arg) {
  return _(&arg->c.b[2]);
}
// CHECK: define dso_local ptr @unit14
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U3:[0-9]+]]
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_I_S:[0-9]+]]
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}

const void *unit15(struct s4 *arg) {
  return _(&arg[2].c.a);
}
// CHECK: define dso_local ptr @unit15
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S4]]
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_I_U]]

const void *unit16(union u3 *arg) {
  return _(&arg[2].a);
}
// CHECK: define dso_local ptr @unit16
// CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%union.u3) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U3]]

// CHECK: ![[POINTER]] = !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK: ![[STRUCT_S4]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s4"
// CHECK: ![[UNION_I_U]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u"
// CHECK: ![[UNION_U3]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u3"
// CHECK: ![[STRUCT_I_S]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s"
// CHECK: ![[STRUCT_S1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1"
// CHECK: ![[STRUCT_S2]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2"
// CHECK: ![[STRUCT_S3]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s3"
// CHECK: ![[UNION_U1]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1"
// CHECK: ![[UNION_U2]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u2"
