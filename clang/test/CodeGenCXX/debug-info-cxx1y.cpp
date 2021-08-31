// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm-only -std=c++14 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// CHECK: imports: [[IMPS:![0-9]*]]

// CHECK: [[IMPS]] = !{[[IMP:![0-9]*]]}
// CHECK: [[IMP]] = !DIImportedEntity(
// CHECK-SAME: entity: [[F3:![0-9]*]]
// CHECK: [[F3]] = distinct !DISubprogram(name: "f3"
// CHECK-SAME:          type: [[SUBROUTINE_TYPE:![0-9]*]]
// CHECK: [[SUBROUTINE_TYPE]] = !DISubroutineType(types: [[TYPE_LIST:![0-9]*]])
// CHECK: [[TYPE_LIST]] = !{[[INT:![0-9]*]]}
// CHECK: [[INT]] = !DIBasicType(name: "int"

// CHECK: [[EMPTY:![0-9]*]] = !{}
// CHECK: [[FOO:![0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo",
// CHECK-SAME:             elements: [[EMPTY]]

// FIXME: The context of this definition should be the CU/file scope, not the class.
// CHECK: !DISubprogram(name: "func", {{.*}} scope: [[FOO]]
// CHECK-SAME:          type: [[SUBROUTINE_TYPE]]
// CHECK-SAME:          DISPFlagDefinition
// CHECK-SAME:          declaration: [[FUNC_DECL:![0-9]*]]
// CHECK: [[FUNC_DECL]] = !DISubprogram(name: "func",
// CHECK-SAME:                          scope: [[FOO]]
// CHECK-SAME:                          type: [[SUBROUTINE_TYPE]]
// CHECK-SAME:                          spFlags: 0

struct foo {
  static auto func();
};

foo f;

auto foo::func() {
  return 1;
}

namespace ns {
auto f2();
auto f3() {
  return 0;
}
}
using ns::f2;
using ns::f3;
