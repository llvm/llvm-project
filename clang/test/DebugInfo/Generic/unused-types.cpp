// RUN: %clang_cc1 -debug-info-kind=unused-types -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck --check-prefix=NODBG %s
using foo = int;
class bar {};
enum class baz { BAZ };

void quux() {
  using x = int;
  class y {};
  enum class z { Z };
}

// CHECK: !DICompileUnit{{.+}}retainedTypes: [[RETTYPES:![0-9]+]]
// CHECK: [[TYPE0:![0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "baz"
// CHECK: [[TYPE1:![0-9]+]] = !DIEnumerator(name: "BAZ"
// CHECK: [[TYPE2:![0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "z"
// CHECK: [[TYPE3:![0-9]+]] = !DIEnumerator(name: "Z"
// CHECK: [[RETTYPES]] = !{[[TYPE4:![0-9]+]], [[TYPE5:![0-9]+]], [[TYPE0]], {{![0-9]+}}, [[TYPE6:![0-9]+]], [[TYPE2]]}
// CHECK: [[TYPE4]] = !DIDerivedType(tag: DW_TAG_typedef, name: "foo"
// CHECK: [[TYPE5]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "bar"
// CHECK: [[TYPE6]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "y"

// NODBG-NOT: !DI{{CompositeType|Enumerator|DerivedType}}

class unused_class;
enum class unused_enum_class;

// NODBG-NOT: name: "unused_
