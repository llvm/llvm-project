// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
typedef struct {
   int a;
} __t;
typedef union {
   int b;
} __u;
#pragma clang attribute pop

int test1(__t *arg) { return arg->a; }
int test2(const __u *arg) { return arg->b; }


// CHECK: define dso_local i32 @test1
// CHECK: call i32* @llvm.preserve.struct.access.index.p0i32.p0s_struct.__ts(%struct.__t* elementtype(%struct.__t) %{{[0-9a-z]+}}, i32 0, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_STRUCT:[0-9]+]]
// CHECK: define dso_local i32 @test2
// CHECK: call %union.__u* @llvm.preserve.union.access.index.p0s_union.__us.p0s_union.__us(%union.__u* %{{[0-9a-z]+}}, i32 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[CONST_TYPEDEF:[0-9]+]]
//
// CHECK: ![[TYPEDEF_STRUCT]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__t"
// CHECK: ![[CONST_TYPEDEF]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[TYPEDEF_UNION:[0-9]+]]
// CHECK: ![[TYPEDEF_UNION]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__u"
