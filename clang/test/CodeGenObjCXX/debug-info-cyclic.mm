// RUN: %clang_cc1 -triple x86_64-apple-darwin -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s

struct B {
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B"
// CHECK-SAME:                             line: [[@LINE-2]],
// CHECK-SAME:                             size: 8, align: 8,
// CHECK-NOT:                              offset:
// CHECK-NOT:                              DIFlagFwdDecl
// CHECK-SAME:                             elements: ![[BMEMBERS:[0-9]+]]
// CHECK-SAME:                             identifier: [[B:.*]])
// CHECK: ![[BMEMBERS]] = !{![[BB:[0-9]+]]}
  B(struct A *);
// CHECK: ![[BB]] = !DISubprogram(name: "B", scope: ![[B]]
// CHECK-SAME:                    line: [[@LINE-2]],
// CHECK-SAME:                    type: ![[TY:[0-9]+]],
// CHECK: ![[TY]] = !DISubroutineType(types: ![[ARGS:[0-9]+]])
// CHECK: ![[ARGS]] = !{null, ![[THIS:[0-9]+]], !{{[^,]+}}}
// CHECK: ![[THIS]] = !DIDerivedType(tag: DW_TAG_pointer_type,
// CHECK-SAME:                       baseType: ![[B]]
};

struct C {
 B b;
 C(struct A *);
 virtual ~C();
};

C::C(struct A *a)
  : b(a) {
}
