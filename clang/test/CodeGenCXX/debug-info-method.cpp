// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -debug-info-kind=limited %s -o - | FileCheck %s
// CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type
// CHECK: ![[A:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A",{{.*}} identifier: "_ZTS1A")
// CHECK: !DISubprogram(name: "foo", linkageName: "_ZN1A3fooEiS_3$_0"
// CHECK-SAME:          DIFlagProtected
// CHECK: ![[THISTYPE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[A]]
// CHECK-SAME:                                  DIFlagArtificial
// CHECK: !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: ![[MEMFUNTYPE:[0-9]+]]
// CHECK: ![[MEMFUNTYPE]] = !DISubroutineType({{(cc: DW_CC_BORLAND_thiscall, )?}}types: ![[MEMFUNARGS:[0-9]+]])
// CHECK: ![[MEMFUNARGS]] = {{.*}}, ![[THISTYPE]],
// CHECK: !DILocalVariable(name: "this", arg: 1
// CHECK: !DILocalVariable(arg: 2
// CHECK: !DILocalVariable(arg: 3
// CHECK: !DILocalVariable(arg: 4
union {
  int a;
  float b;
} u;

class A {
protected:
  void foo(int, A, decltype(u));
}; 

void A::foo(int, A, decltype(u)) {
}

A a;

int A::*x = 0;
int (A::*y)(int) = 0;
