// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -std=c++2b %s -o - | FileCheck %s

struct Foo {
  void Bar(this Foo&& self) {}
};

void fn() {
  Foo{}.Bar();
}

// CHECK: distinct !DISubprogram(name: "Bar", {{.*}}, type: ![[BAR_TYPE:[0-9]+]], {{.*}}, declaration: ![[BAR_DECL:[0-9]+]], {{.*}}
// CHECK: ![[FOO:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo"
// CHECK: ![[BAR_DECL]] = !DISubprogram(name: "Bar", {{.*}}, type: ![[BAR_TYPE]], {{.*}},
// CHECK: ![[BAR_TYPE]] = !DISubroutineType(types: ![[PARAMS:[0-9]+]])
// CHECK: ![[PARAMS]] = !{null, ![[SELF:[0-9]+]]}
// CHECK: ![[SELF]] = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: ![[FOO]]
