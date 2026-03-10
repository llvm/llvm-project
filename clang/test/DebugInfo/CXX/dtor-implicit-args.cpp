// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -debug-info-kind=limited %s -o - | FileCheck --check-prefix MSVC %s

struct B {
    virtual ~B() {}
};

struct A : virtual B {
};

A a;


// CHECK-DAG: !{{[0-9]+}} = !DILocalVariable(name: "vtt", arg: 2, scope: ![[destructor:[0-9]+]], type: ![[vtttype:[0-9]+]], flags: DIFlagArtificial)
// CHECK-DAG: ![[destructor]] = distinct !DISubprogram(name: "~A", {{.*}}, type: ![[subroutinetype:[0-9]+]]
// CHECK-DAG: ![[subroutinetype]] = !DISubroutineType(types: ![[types:[0-9]+]])
// CHECK-DAG: [[types]] = !{null, !{{[0-9]+}}, ![[vtttype]]}

// MSVC-DAG: ![[inttype:[0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// MSVC-DAG: ![[voidpointertype:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// MSVC-DAG: ![[destructor:[0-9]+]] = distinct !DISubprogram(name: "~A", linkageName: "??_GA@@UEAAPEAXI@Z", {{.*}}, type: ![[subroutinetype:[0-9]+]]
// MSVC-DAG: !{{[0-9]+}} = !DILocalVariable(name: "should_call_delete", arg: 2, scope: ![[destructor]], type: ![[inttype]], flags: DIFlagArtificial)
// MSVC-DAG: ![[subroutinetype]] = !DISubroutineType(types: ![[types:[0-9]+]])
// MSVC-DAG: [[types]] = !{![[voidpointertype]], !{{[0-9]+}}, ![[inttype]]}
