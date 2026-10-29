// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -debug-info-kind=standalone -gtemplate-alias %s | FileCheck %s

namespace test1 {
  struct A {};
  template <class> using B = A;
  template <class T> struct C : B<T> {};
  C<int> t1;

  // CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "C<int>", {{.+}}, elements: ![[test1_bases:[0-9]+]], {{.+}}, identifier: "_ZTSN5test11CIiEE"
  // CHECK-NEXT: ![[test1_bases]] = !{![[test1_inheritance:[0-9]+]]}
  // CHECK-NEXT: ![[test1_inheritance]] = !DIDerivedType(tag: DW_TAG_inheritance, scope: ![[#]], baseType: ![[test1_alias:[0-9]+]],
  // CHECK-NEXT: ![[test1_alias]] = !DIDerivedType(tag: DW_TAG_template_alias, name: "B<int>", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[test1_basetype:[0-9]+]],
  // CHECK-NEXT: ![[test1_basetype]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", {{.+}} identifier: "_ZTSN5test11AE"
} // namespace test1
