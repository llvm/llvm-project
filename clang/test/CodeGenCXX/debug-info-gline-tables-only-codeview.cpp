// RUN: %clang_cc1 %s -gcodeview -debug-info-kind=line-tables-only -S \
// RUN:   -emit-llvm -o - | FileCheck %s
// Checks that clang with "-gline-tables-only" with CodeView emits some debug
// info for variables and types when they appear in function scopes.

namespace NS {
struct C {
  void m() {}
};
void f() {}
}

NS::C c;

void test() {
  // CHECK: !DISubprogram(name: "f", scope: ![[NS:[0-9]+]],
  // CHECK-SAME:          type: ![[F:[0-9]+]]
  // CHECK: ![[NS]] = !DINamespace(name: "NS", scope: null)
  // CHECK: ![[F]] = !DISubroutineType(types: ![[FTYPE:[0-9]+]])
  // CHECK: ![[FTYPE]] = !{null}
  NS::f();

  // CHECK: ![[M:[0-9]+]] = distinct !DISubprogram(name: "m", scope: ![[C:[0-9]+]],
  // CHECK-SAME:                                   type: ![[MTYPE:[0-9]+]]
  // CHECK: ![[C]] = !DICompositeType(tag: DW_TAG_structure_type, name: "C",
  // CHECK-SAME:                      flags: DIFlagFwdDecl
  // CHECK-NOT: identifier
  // CHECK: ![[MTYPE]] = !DISubroutineType({{.*}}types: !{{.*}})
  c.m();
}
