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
// CHECK: [[RETTYPES]] = !{[[TYPE2:![0-9]+]], [[TYPE3:![0-9]+]], [[TYPE0]], {{![0-9]+}}}
// CHECK: [[TYPE2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "foo"
// CHECK: [[TYPE3]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "bar"
// CHECK: [[SP:![0-9]+]] = distinct !DISubprogram(name: "quux", {{.*}}, retainedNodes: [[SPRETNODES:![0-9]+]]
// CHECK: [[SPRETNODES]] = !{[[TYPE4:![0-9]+]], [[TYPE5:![0-9]+]]}
// CHECK: [[TYPE4]] = distinct !DICompositeType(tag: DW_TAG_class_type, name: "y", scope: [[SP]]
// CHECK: [[TYPE5]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "z", scope: [[SP]]
// CHECK: [[TYPE6:![0-9]+]] = !DIEnumerator(name: "Z"

// NODBG-NOT: !DI{{CompositeType|Enumerator|DerivedType}}

class unused_class;
enum class unused_enum_class;

// NODBG-NOT: name: "unused_
