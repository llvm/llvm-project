// RUN: %clang_cc1 -triple x86_64-windows -debug-info-kind=limited -gcodeview %s -emit-llvm -o - | FileCheck %s

// Test member pointer inheritance models.

struct A { int a; };
struct B { int b; };
struct C : A, B { int c; };
struct D : virtual C { int d; };
struct E;
int A::*pmd_a;
int C::*pmd_b;
int D::*pmd_c;
int E::*pmd_d;
void (A::*pmf_a)();
void (C::*pmf_b)();
void (D::*pmf_c)();
void (E::*pmf_d)();

// Test incomplete MPTs, which don't have inheritance models.

struct Incomplete;
int Incomplete::**ppmd;
void (Incomplete::**ppmf)();

// CHECK: distinct !DIGlobalVariable(name: "pmd_a", {{.*}} type: ![[pmd_a:[^, ]*]], {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmd_b", {{.*}} type: ![[pmd_b:[^, ]*]], {{.*}})
// CHECK: ![[pmd_b]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 32, flags: DIFlagMultipleInheritance, {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmd_c", {{.*}} type: ![[pmd_c:[^, ]*]], {{.*}})
// CHECK: ![[pmd_c]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 64, flags: DIFlagVirtualInheritance, {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmd_d", {{.*}} type: ![[pmd_d:[^, ]*]], {{.*}})
// CHECK: ![[pmd_d]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 96,
// CHECK-NOT: flags:
// CHECK-SAME: ){{$}}

// CHECK: distinct !DIGlobalVariable(name: "pmf_a", {{.*}} type: ![[pmf_a:[^, ]*]], {{.*}})
// CHECK: ![[pmf_a]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 64, flags: DIFlagSingleInheritance, {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmf_b", {{.*}} type: ![[pmf_b:[^, ]*]], {{.*}})
// CHECK: ![[pmf_b]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 128, flags: DIFlagMultipleInheritance, {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmf_c", {{.*}} type: ![[pmf_c:[^, ]*]], {{.*}})
// CHECK: ![[pmf_c]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 128, flags: DIFlagVirtualInheritance, {{.*}})
// CHECK: distinct !DIGlobalVariable(name: "pmf_d", {{.*}} type: ![[pmf_d:[^, ]*]], {{.*}})
// CHECK: ![[pmf_d]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 192,
// CHECK-NOT: flags:
// CHECK-SAME: ){{$}}

// CHECK: distinct !DIGlobalVariable(name: "ppmd", {{.*}} type: ![[ppmd:[^, ]*]], {{.*}})
// CHECK: ![[ppmd]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[ppmd2:[^ ]*]], size: 64, align: 64)
// CHECK: ![[ppmd2]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{[0-9]*}}, extraData: !{{[0-9]*}}){{$}}
// CHECK: distinct !DIGlobalVariable(name: "ppmf", {{.*}} type: ![[ppmf:[^, ]*]], {{.*}})
// CHECK: ![[ppmf]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[ppmf2:[^ ]*]], size: 64, align: 64)
// CHECK: ![[ppmf2]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{[0-9]*}}, extraData: !{{[0-9]*}}){{$}}

// CHECK: ![[pmd_a]] = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !{{.*}}, size: 32, flags: DIFlagSingleInheritance, {{.*}})
