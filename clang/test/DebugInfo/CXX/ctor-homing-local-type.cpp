// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -debug-info-kind=limited -dwarf-version=5 -O0 -disable-llvm-passes %s -o - \
// RUN:        | FileCheck %s

// When compiling this with limited debug info, a replaceable forward declaration DICompositeType
// for "n" is created in the scope of base object constructor (C2) of class l, and it's not immediately
// replaced with a distinct node (the type is not "completed").
// Later, it gets replaced with distinct definition DICompositeType, which is created in the scope of
// complete object constructor (C1).
//
// In contrast to that, in standalone debug info mode, the complete definition DICompositeType
// for "n" is created sooner, right in the context of C2.
//
// Check that DIBuilder processes the limited debug info case correctly, and doesn't add the same
// local type to retainedNodes fields of both DISubprograms (C1 and C2).

// CHECK: ![[C2:[0-9]+]] = distinct !DISubprogram(name: "l", linkageName: "_ZN1lC2Ev", {{.*}}, retainedNodes: ![[EMPTY:[0-9]+]])
// CHECK: ![[EMPTY]] = !{}
// CHECK: ![[N:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "n",
// CHECK: ![[C1:[0-9]+]] = distinct !DISubprogram(name: "l", linkageName: "_ZN1lC1Ev", {{.*}}, retainedNodes: ![[RN:[0-9]+]])
// CHECK: ![[RN]] = !{![[N]]}

template <class d>
struct k {
  void i() {
    new d;
  }
};

struct l {
  l();
};

l::l() {
  struct n {};
  k<n> m;
  m.i();
}
