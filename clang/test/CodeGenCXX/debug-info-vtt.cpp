// RUN: %clang -g -c -emit-llvm %s -o - | llvm-dis | FileCheck %s

struct B {
    virtual ~B() {}
};

struct A : virtual B {
};

A a;

// CHECK-DAG: distinct !DISubprogram(name: "~A", linkageName: "_ZN1AD2Ev", {{.*}}, type: ![[subroutinetype:[0-9]+]]
// CHECK-DAG: ![[subroutinetype]] = !DISubroutineType(types: ![[types:[0-9]+]])
// CHECK-DAG: [[types]] = !{null, !{{[0-9]+}}, !{{[0-9]+}}}


