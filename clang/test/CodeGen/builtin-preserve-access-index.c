// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

#define _(x) (__builtin_preserve_access_index(x))

const void *unit1(const void *arg) {
  return _(arg);
}
// CHECK: define dso_local i8* @unit1
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit2(void) {
  return _((const void *)0xffffffffFFFF0000ULL);
}
// CHECK: define dso_local i8* @unit2
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit3(const int *arg) {
  return _(arg + 1);
}
// CHECK: define dso_local i8* @unit3
// CHECK-NOT: llvm.preserve.array.access.index
// CHECK-NOT: llvm.preserve.struct.access.index
// CHECK-NOT: llvm.preserve.union.access.index

const void *unit4(const int *arg) {
  return _(&arg[1]);
}
// CHECK: define dso_local i8* @unit4
// CHECK-NOT: getelementptr
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* elementtype(i32) %{{[0-9a-z]+}}, i32 0, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[POINTER:[0-9]+]]

const void *unit5(const int *arg[5]) {
  return _(&arg[1][2]);
}
// CHECK: define dso_local i8* @unit5
// CHECK-NOT: getelementptr
// CHECK: call i32** @llvm.preserve.array.access.index.p0p0i32.p0p0i32(i32** elementtype(i32*) %{{[0-9a-z]+}}, i32 0, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK-NOT: getelementptr
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* elementtype(i32) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[POINTER:[0-9]+]]

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
// CHECK: define dso_local i8* @unit6
// CHECK-NOT: getelementptr
// CHECK: call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]

const void *unit7(struct s1 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local i8* @unit7
// CHECK-NOT: getelementptr
// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1]]

const void *unit8(struct s2 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local i8* @unit8
// CHECK-NOT: getelementptr
// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s2s(%struct.s2* elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S2:[0-9]+]]

const void *unit9(struct s3 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local i8* @unit9
// CHECK-NOT: getelementptr
// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.s3s(%struct.s3* elementtype(%struct.s3) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S3:[0-9]+]]

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
// CHECK: define dso_local i8* @unit10
// CHECK-NOT: getelementptr
// CHECK: call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1:[0-9]+]]

const void *unit11(union u1 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local i8* @unit11
// CHECK-NOT: getelementptr
// CHECK: call %union.u1* @llvm.preserve.union.access.index.p0s_union.u1s.p0s_union.u1s(%union.u1* %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1]]

const void *unit12(union u2 *arg) {
  return _(&arg->b);
}
// CHECK: define dso_local i8* @unit12
// CHECK-NOT: getelementptr
// CHECK: call %union.u2* @llvm.preserve.union.access.index.p0s_union.u2s.p0s_union.u2s(%union.u2* %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U2:[0-9]+]]

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
// CHECK: define dso_local i8* @unit13
// CHECK: call %union.u* @llvm.preserve.struct.access.index.p0s_union.us.p0s_struct.s4s(%struct.s4* elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S4:[0-9]+]]
// CHECK: call %union.u* @llvm.preserve.union.access.index.p0s_union.us.p0s_union.us(%union.u* %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_I_U:[0-9]+]]
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]* elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}

const void *unit14(union u3 *arg) {
  return _(&arg->c.b[2]);
}
// CHECK: define dso_local i8* @unit14
// CHECK: call %union.u3* @llvm.preserve.union.access.index.p0s_union.u3s.p0s_union.u3s(%union.u3* %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U3:[0-9]+]]
// CHECK: call [4 x i32]* @llvm.preserve.struct.access.index.p0a4i32.p0s_struct.ss(%struct.s* elementtype(%struct.s) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_I_S:[0-9]+]]
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]* elementtype([4 x i32]) %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}

const void *unit15(struct s4 *arg) {
  return _(&arg[2].c.a);
}
// CHECK: define dso_local i8* @unit15
// CHECK: call %struct.s4* @llvm.preserve.array.access.index.p0s_struct.s4s.p0s_struct.s4s(%struct.s4* elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK: call %union.u* @llvm.preserve.struct.access.index.p0s_union.us.p0s_struct.s4s(%struct.s4* elementtype(%struct.s4) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S4]]
// CHECK: call %union.u* @llvm.preserve.union.access.index.p0s_union.us.p0s_union.us(%union.u* %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_I_U]]

const void *unit16(union u3 *arg) {
  return _(&arg[2].a);
}
// CHECK: define dso_local i8* @unit16
// CHECK: call %union.u3* @llvm.preserve.array.access.index.p0s_union.u3s.p0s_union.u3s(%union.u3* elementtype(%union.u3) %{{[0-9a-z]+}}, i32 0, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index !{{[0-9]+}}
// CHECK: call %union.u3* @llvm.preserve.union.access.index.p0s_union.u3s.p0s_union.u3s(%union.u3* %{{[0-9a-z]+}}, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U3]]

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
