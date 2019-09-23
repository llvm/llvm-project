// RUN: %clang -target x86_64 -emit-llvm -S -g %s -o - | FileCheck %s

#define _(x) (__builtin_preserve_access_index(x))

struct s1 {
  char a;
  int b[4];
};

int unit1(struct s1 *arg) {
  return _(arg->b[2]);
}
// CHECK: define dso_local i32 @unit1
// CHECK: call [4 x i32]* @llvm.preserve.struct.access.index.p0a4i32.p0s_struct.s1s(%struct.s1* %{{[0-9a-z]+}}, i32 1, i32 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S1:[0-9]+]]
// CHECK: call i32* @llvm.preserve.array.access.index.p0i32.p0a4i32([4 x i32]* %{{[0-9a-z]+}}, i32 1, i32 2), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ARRAY:[0-9]+]]
//
// CHECK: ![[ARRAY]] = !DICompositeType(tag: DW_TAG_array_type
// CHECK: ![[STRUCT_S1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s1"
