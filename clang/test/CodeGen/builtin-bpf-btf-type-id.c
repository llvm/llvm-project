// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g %s -o - | FileCheck %s

unsigned test1(int a) { return __builtin_btf_type_id(a, 0); }
unsigned test2(int a) { return __builtin_btf_type_id(&a, 0); }

// CHECK: define dso_local i32 @test1
// CHECK: call i32 @llvm.bpf.btf.type.id.p0i32.i32(i32* %{{[0-9a-z.]+}}, i32 1, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[INT:[0-9]+]]
// CHECK: define dso_local i32 @test2
// CHECK: call i32 @llvm.bpf.btf.type.id.p0i32.i32(i32* %{{[0-9a-z.]+}}, i32 0, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[INT_POINTER:[0-9]+]]
//
// CHECK: ![[INT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed
// CHECK: ![[INT_POINTER]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[INT]], size: 64
