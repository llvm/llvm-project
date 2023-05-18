// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

struct HasNested {
  enum InnerEnum { _BUF_SIZE = 1 };
  typedef int InnerTypedef;
  enum { InnerEnumerator = 2 };
  struct InnerStruct { };

  union { int i1; };
  union { int i2; } unnamed_union;
  struct { int i3; };
  struct { int i4; } unnamed_struct;
};
HasNested f;

// CHECK: ![[INNERENUM:[0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "InnerEnum", {{.*}})
// CHECK: ![[HASNESTED:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasNested",
// CHECK-SAME: elements: ![[MEMBERS:[0-9]+]],
//
// CHECK: ![[MEMBERS]] = !{![[INNERENUM]], ![[INNERTYPEDEF:[0-9]+]], ![[UNNAMEDENUM:[0-9]+]], ![[INNERSTRUCT:[0-9]+]], ![[ANONYMOUS_UNION:[0-9]+]], ![[UNNAMED_UNION_TYPE:[0-9]+]], ![[UNNAMED_UNION:[0-9]+]], ![[ANONYMOUS_STRUCT:[0-9]+]], ![[UNNAMED_STRUCT_TYPE:[0-9]+]], ![[UNNAMED_STRUCT:[0-9]+]]}
//
// CHECK: ![[INNERTYPEDEF]] = !DIDerivedType(tag: DW_TAG_typedef, name: "InnerTypedef", scope: ![[HASNESTED]]{{.*}})
//
// CHECK: ![[UNNAMEDENUM]] = !DICompositeType(tag: DW_TAG_enumeration_type, scope: ![[HASNESTED]],
// CHECK-SAME: elements: ![[UNNAMEDMEMBERS:[0-9]+]],
// CHECK: ![[UNNAMEDMEMBERS]] = !{![[INNERENUMERATOR:[0-9]+]]}
// CHECK: ![[INNERENUMERATOR]] = !DIEnumerator(name: "InnerEnumerator", value: 2)
//
// CHECK: ![[INNERSTRUCT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "InnerStruct"
// CHECK-SAME: flags: DIFlagFwdDecl

// CHECK: ![[ANONYMOUS_UNION]] = !DIDerivedType(tag: DW_TAG_member, 
// CHECK-SAME: baseType: ![[ANONYMOUS_UNION_TYPE:[0-9]+]]
// CHECK: ![[ANONYMOUS_UNION_TYPE]] = distinct !DICompositeType(tag: DW_TAG_union_type,
// CHECK-SAME: elements: ![[ANONYMOUS_UNION_MEMBERS:[0-9]+]], identifier: ".?AT<unnamed-type-$S1>@HasNested@@")
// CHECK: ![[ANONYMOUS_UNION_MEMBERS]] = !{![[ANONYMOUS_UNION_MEMBER:[0-9]+]]}
// CHECK: ![[ANONYMOUS_UNION_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "i1"

// CHECK: ![[UNNAMED_UNION_TYPE]] = distinct !DICompositeType(tag: DW_TAG_union_type, name: "<unnamed-type-unnamed_union>"
// CHECK-SAME: elements: ![[UNNAMED_UNION_MEMBERS:[0-9]+]], identifier: ".?AT<unnamed-type-unnamed_union>@HasNested@@")
// CHECK: ![[UNNAMED_UNION_MEMBERS]] = !{![[UNNAMED_UNION_MEMBER:[0-9]+]]}
// CHECK: ![[UNNAMED_UNION_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "i2"
// CHECK: ![[UNNAMED_UNION]] = !DIDerivedType(tag: DW_TAG_member, name: "unnamed_union"
// CHECK-SAME: baseType: ![[UNNAMED_UNION_TYPE]]

// CHECK: ![[ANONYMOUS_STRUCT]] = !DIDerivedType(tag: DW_TAG_member
// CHECK-SAME: baseType: ![[ANONYMOUS_STRUCT_TYPE:[0-9]+]]
// CHECK: ![[ANONYMOUS_STRUCT_TYPE]] = distinct !DICompositeType(tag: DW_TAG_structure_type
// CHECK-SAME: elements: ![[ANONYMOUS_STRUCT_MEMBERS:[0-9]+]], identifier: ".?AU<unnamed-type-$S2>@HasNested@@")
// CHECK: ![[ANONYMOUS_STRUCT_MEMBERS]] = !{![[ANONYMOUS_STRUCT_MEMBER:[0-9]+]]}
// CHECK: ![[ANONYMOUS_STRUCT_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "i3"

// CHECK: ![[UNNAMED_STRUCT_TYPE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "<unnamed-type-unnamed_struct>"
// CHECK-SAME: elements: ![[UNNAMED_STRUCT_MEMBERS:[0-9]+]], identifier: ".?AU<unnamed-type-unnamed_struct>@HasNested@@")
// CHECK: ![[UNNAMED_STRUCT_MEMBERS]] = !{![[UNNAMED_STRUCT_MEMBER:[0-9]+]]}
// CHECK: ![[UNNAMED_STRUCT_MEMBER]] = !DIDerivedType(tag: DW_TAG_member, name: "i4"
// CHECK: ![[UNNAMED_STRUCT]] = !DIDerivedType(tag: DW_TAG_member, name: "unnamed_struct"
// CHECK-SAME: baseType: ![[UNNAMED_STRUCT_TYPE]]
