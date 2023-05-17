// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g -Xclang -disable-llvm-passes %s -o - | FileCheck %s

unsigned test1(int a) { return __builtin_btf_type_id(a, 0); }
unsigned test2(int a) { return __builtin_btf_type_id(&a, 0); }

struct t1 { int a; };
typedef struct t1 __t1;
unsigned test3() {
  return __builtin_btf_type_id(*(struct t1 *)0, 1) +
         __builtin_btf_type_id(*(__t1 *)0, 1);
}

// CHECK: define dso_local i32 @test1
// CHECK: call i64 @llvm.bpf.btf.type.id(i32 0, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[INT:[0-9]+]]
// CHECK: define dso_local i32 @test2
// CHECK: call i64 @llvm.bpf.btf.type.id(i32 1, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[INT_POINTER:[0-9]+]]
// CHECK: define dso_local i32 @test3
// CHECK: call i64 @llvm.bpf.btf.type.id(i32 2, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[STRUCT_T1:[0-9]+]]
// CHECK: call i64 @llvm.bpf.btf.type.id(i32 3, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_T1:[0-9]+]]
//
// CHECK: ![[INT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed
// CHECK: ![[INT_POINTER]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[INT]], size: 64
// CHECK: ![[STRUCT_T1]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1"
// CHECK: ![[TYPEDEF_T1]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__t1"
