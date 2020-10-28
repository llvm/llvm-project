// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g -Xclang -disable-llvm-passes %s -o - | FileCheck %s

#define _(x, y) (__builtin_preserve_type_info((x), (y)))

struct s {
  char a;
};
typedef int __int;
enum AA {
  VAL1 = 1,
  VAL2 = 2,
};

unsigned unit1() {
  struct s v = {};
  return _(v, 0) + _(*(struct s *)0, 0);
}

// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 0, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 1, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_S]]

unsigned unit2() {
  __int n;
  return _(n, 1) + _(*(__int *)0, 1);
}

// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 2, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_INT:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 3, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_INT]]

unsigned unit3() {
  enum AA t;
  return _(t, 0) + _(*(enum AA *)0, 1);
}

// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 4, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM_AA:[0-9]+]]
// CHECK: call i32 @llvm.bpf.preserve.type.info(i32 5, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM_AA]]

// CHECK: ![[ENUM_AA]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA"
// CHECK: ![[TYPEDEF_INT]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__int"
// CHECK: ![[STRUCT_S]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s"
