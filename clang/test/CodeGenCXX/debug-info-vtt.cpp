// RUN: %clang -g -c -emit-llvm %s -o - | llvm-dis | FileCheck %s

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
