// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g -Xclang -disable-llvm-passes %s -o - | FileCheck %s

#define _(x, y) (__builtin_preserve_field_info((x), (y)))

struct s1 {
  char a;
  char b:2;
};
struct s2 {
  struct s1 s;
};

unsigned unit1(struct s2 *arg) {
  return _(arg->s.a, 10) + _(arg->s.b, 10);
}
// CHECK: define dso_local i32 @unit1
// CHECK: call %struct.s1* @llvm.preserve.struct.access.index.p0s_struct.s1s.p0s_struct.s2s(%struct.s2* elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S2:[0-9]+]]
// CHECK: call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}
// CHECK: call %struct.s1* @llvm.preserve.struct.access.index.p0s_struct.s1s.p0s_struct.s2s(%struct.s2* elementtype(%struct.s2) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S2:[0-9]+]]
// CHECK: call i8* @llvm.preserve.struct.access.index.p0i8.p0s_struct.s1s(%struct.s1* elementtype(%struct.s1) %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.field.info.p0i8(i8* %{{[0-9a-z]+}}, i64 10), !dbg !{{[0-9]+}}

// CHECK: ![[STRUCT_S2]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2"
// CHECK: ![[STRUCT_S1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1"
