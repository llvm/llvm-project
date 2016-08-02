// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only -std=c++14 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: [[EMPTY:![0-9]*]] = !{}
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo",
// CHECK-SAME:             elements: [[EMPTY]]
// FIXME: The context of this definition should be the CU/file scope, not the class.
// CHECK: !DISubprogram(name: "func", {{.*}} scope: !"_ZTS3foo"
// CHECK-SAME:          type: [[SUBROUTINE_TYPE:![0-9]*]]
// CHECK-SAME:          isDefinition: true
// CHECK-SAME:          declaration: [[FUNC_DECL:![0-9]*]]
// CHECK: [[SUBROUTINE_TYPE]] = !DISubroutineType(types: [[TYPE_LIST:![0-9]*]])
// CHECK: [[TYPE_LIST]] = !{[[INT:![0-9]*]]}
// CHECK: [[INT]] = !DIBasicType(name: "int"
// CHECK: [[FUNC_DECL]] = !DISubprogram(name: "func",
// CHECK-SAME:                          scope: !"_ZTS3foo"
// CHECK-SAME:                          type: [[SUBROUTINE_TYPE]]
// CHECK-SAME:                          isDefinition: false

struct foo {
  static auto func();
};

foo f;

auto foo::func() {
  return 1;
}
