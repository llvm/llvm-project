// RUN: %clang_cc1 %s -triple x86_64-windows-msvc -gcodeview \
// RUN:   -debug-info-kind=line-tables-only -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" with CodeView emits some debug
// info for variables and types when they appear in function scopes.

namespace NS {
struct C {
  void m() {}
  // Test externally visible lambda.
  void lambda2() { []() {}(); }
 
  // Test naming for function parameters.
  void lambda_params(int x = [](){ return 0; }(), int y = [](){ return 1; }()) {}
};
void f() {}
}

// Test non- externally visible lambda.
auto lambda1 = []() { return 1; };

NS::C c;


void test() {
  // CHECK: !DISubprogram(name: "f", scope: ![[NS:[0-9]+]],
  // CHECK-SAME:          type: ![[F:[0-9]+]]
  // CHECK: ![[NS]] = !DINamespace(name: "NS", scope: null)
  // CHECK: ![[F]] = !DISubroutineType(types: ![[FTYPE:[0-9]+]])
  // CHECK: ![[FTYPE]] = !{null}
  NS::f();

  // CHECK: ![[M:[0-9]+]] = distinct !DISubprogram(name: "m", scope: ![[C:[0-9]+]],
  // CHECK-SAME:                                   type: ![[MTYPE:[0-9]+]],
  // CHECK: ![[C]] = !DICompositeType(tag: DW_TAG_structure_type, name: "C",
  // CHECK-SAME:                      flags: DIFlagFwdDecl
  // CHECK-NOT: identifier
  // CHECK: ![[MTYPE]] = !DISubroutineType({{.*}}types: !{{.*}})
  c.m();

  // CHECK: !DISubprogram(name: "operator()", scope: ![[LAMBDA0:[0-9]+]],
  // CHECK: ![[LAMBDA0]] = !DICompositeType(tag: DW_TAG_class_type,
  // CHECK-SAME:                            name: "<lambda_0>",
  // CHECK-SAME:                            flags: DIFlagFwdDecl
  lambda1();

  // CHECK: !DISubprogram(name: "operator()", scope: ![[LAMBDA1_1:[0-9]+]],
  // CHECK: ![[LAMBDA1_1]] = !DICompositeType(tag: DW_TAG_class_type,
  // CHECK-SAME:                              name: "<lambda_1_1>",
  // CHECK: !DISubprogram(name: "operator()", scope: ![[LAMBDA2_1:[0-9]+]],
  // CHECK: ![[LAMBDA2_1]] = !DICompositeType(tag: DW_TAG_class_type,
  // CHECK-SAME:                              name: "<lambda_2_1>",
  c.lambda_params();

  // CHECK: ![[LAMBDA1:[0-9]+]] = !DICompositeType(tag: DW_TAG_class_type,
  // CHECK-SAME:                            name: "<lambda_1>",
  // CHECK-SAME:                            flags: DIFlagFwdDecl
  // CHECK: !DISubprogram(name: "operator()", scope: ![[LAMBDA1]],
  c.lambda2();
}
