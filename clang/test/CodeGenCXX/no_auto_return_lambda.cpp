// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// We emit "auto" for deduced return types for member functions but we should
// not emitting "auto" for deduced return types for lambdas call function which
// will be implmented as operator() in a class type. This test will verify that
// behavior.

__attribute__((used)) int g() {
  auto f = []() { return 10; };
  return f();
}

// g() is not a member function so we should not emit "auto" for the deduced
// return type.
//
// CHECK: !DISubprogram(name: "g",{{.*}}, type: ![[FUN_TYPE:[0-9]+]],{{.*}}
// CHECK: ![[FUN_TYPE]] = !DISubroutineType(types: ![[TYPE_NODE:[0-9]+]])
// CHECK: ![[TYPE_NODE]] = !{![[INT_TYPE:[0-9]+]]}
// CHECK: ![[INT_TYPE]] = !DIBasicType(name: "int", {{.*}})

// operator() of the local lambda should have the same return type as g()
//
// CHECK: distinct !DISubprogram(name: "operator()",{{.*}}, type: ![[FUN_TYPE_LAMBDA:[0-9]+]],{{.*}}
// CHECK: ![[FUN_TYPE_LAMBDA]] = !DISubroutineType({{.*}}types: ![[TYPE_NODE_LAMBDA:[0-9]+]])
// CHECK: ![[TYPE_NODE_LAMBDA]] = !{![[INT_TYPE]], {{.*}}
