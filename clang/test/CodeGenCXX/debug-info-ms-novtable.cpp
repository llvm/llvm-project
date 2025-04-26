// RUN: %clang -g -S -emit-llvm -fms-extensions -fms-compatibility -target x86_64-pc-windows-msvc -o - %s | FileCheck %s

// CHECK-DAG: ![[FOO:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Foo", file: !{{[0-9]+}}, line: {{[0-9]+}}, size: 128, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: ![[FOO_ELEMENTS:[0-9]+]], vtableHolder: ![[FOO]], identifier: ".?AVFoo@@")
// CHECK-DAG: ![[FOO_ELEMENTS]] = !{![[FOO_VTBL_TY:[0-9]+]], ![[FOO_VTBL_MEMBER:[0-9]+]], ![[FOO_MEMBER:[0-9]+]], ![[FOO_DUMMY:[0-9]+]]}
// CHECK-DAG: ![[FOO_VTBL_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 64)
// CHECK-DAG: ![[FOO_VTBL_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$Foo", scope: !10, file: !{{[0-9]+}}, baseType: ![[FOO_VTBL_PTR_TY:[0-9]+]], size: 64, flags: DIFlagArtificial)
// CHECK-DAG: ![[FOO_VTBL_PTR_TY]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[FOO_VTBL_TY]], size: 64)
// CHECK-DAG: ![[FOO_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "member", scope: ![[FOO]], file: !{{[0-9]+}}, line: {{[0-9]+}}, baseType: !{{[0-9]+}}, size: 32, offset: 64)
// CHECK-DAG: ![[FOO_DUMMY]] = !DISubprogram(name: "dummy", linkageName: "?dummy@Foo@@EEAAXXZ", scope: ![[FOO]], file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}}, scopeLine: {{[0-9]+}}, containingType: ![[FOO]], virtualIndex: 0, flags: DIFlagPrototyped | DIFlagIntroducedVirtual, spFlags: DISPFlagVirtual)
class __declspec(novtable) Foo {
    virtual void dummy() noexcept {};

    int member = 1;
};

void foo(Foo) {}
