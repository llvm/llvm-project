// RUN: %clang_cc1 -debug-info-kind=unused-types -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck --check-prefix=NODBG %s
typedef int my_int;
struct foo {};
enum bar { BAR };
union baz {};

void quux(void) {
  typedef int x;
  struct y {};
  enum z { Z };
  union w {};
}

// Check that debug info is emitted for the typedef, struct, enum, and union
// when -fno-eliminate-unused-debug-types and -g are set.

// CHECK: !DICompileUnit{{.+}}retainedTypes: [[RETTYPES:![0-9]+]]
// CHECK: [[TYPE0:![0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "bar"
// CHECK: [[TYPE1:![0-9]+]] = !DIEnumerator(name: "BAR"
// CHECK: [[RETTYPES]] = !{[[TYPE2:![0-9]+]], [[TYPE3:![0-9]+]], [[TYPE0]], [[TYPE4:![0-9]+]], {{![0-9]+}}}
// CHECK: [[TYPE2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "my_int"
// CHECK: [[TYPE3]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK: [[TYPE4]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "baz"
// CHECK: [[SP:![0-9]+]] = distinct !DISubprogram(name: "quux", {{.*}}, retainedNodes: [[SPRETNODES:![0-9]+]]
// CHECK: [[SPRETNODES]] = !{[[TYPE5:![0-9]+]], [[TYPE6:![0-9]+]], [[TYPE8:![0-9]+]]}
// CHECK: [[TYPE5]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "y"
// CHECK: [[TYPE6]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "z"
// CHECK: [[TYPE7:![0-9]+]] = !DIEnumerator(name: "Z"
// CHECK: [[TYPE8]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "w"

// Check that debug info is not emitted for the typedef, struct, enum, and
// union when -fno-eliminate-unused-debug-types and -g are not set. These are
// the same checks as above with `NODBG-NOT` rather than `CHECK`.

// NODBG-NOT: !DI{{CompositeType|Enumerator|DerivedType}}

// Check that debug info is not emitted for declarations. Obnoxious
// indentifiers are to avoid collisions with the SHA emittied as debug info.
struct unused_struct;
enum unused_enum;
union unused_union;
void b0(void) {
  struct unused_local_struct;
  enum unused_local_enum;
  union unused_local_union;
}

// NODBG-NOT: name: "unused_
