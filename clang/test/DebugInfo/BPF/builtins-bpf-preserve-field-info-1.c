// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#define _(x, y) (__builtin_preserve_field_info((x), (y)))

struct s1 {
  char a;
  char b:2;
};

union u1 {
  char a;
  char b:2;
};

unsigned unit1(struct s1 *arg) {
  return _(arg->a, 10) + _(arg->b, 10);
}
// CHECK: define dso_local i32 @unit1
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0(ptr %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0(ptr %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}

unsigned unit2(union u1 *arg) {
  return _(arg->a, 10) + _(arg->b, 10);
}
// CHECK: define dso_local i32 @unit2
// CHECK: call ptr @llvm.preserve.union.access.index.p0.p0(ptr %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0(ptr %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}
// CHECK: call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%union.u1) %{{[0-9a-z]+}}, i32 0, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[UNION_U1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0(ptr %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}

// CHECK: ![[STRUCT_S1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1"
// CHECK: ![[UNION_U1]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "u1"
