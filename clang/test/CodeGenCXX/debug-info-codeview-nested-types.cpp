// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-pc-windows-msvc -debug-info-kind=limited -gcodeview -emit-llvm -o - | FileCheck %s

struct HasNested {
  enum InnerEnum { _BUF_SIZE = 1 };
  typedef int InnerTypedef;
  enum { InnerEnumerator = 2 };
  struct InnerStruct { };
};
HasNested f;

// CHECK: ![[INNERENUM:[0-9]+]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "InnerEnum", {{.*}})
// CHECK: ![[HASNESTED:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "HasNested",
// CHECK-SAME: elements: ![[MEMBERS:[0-9]+]],
//
// CHECK: ![[MEMBERS]] = !{![[INNERENUM]], ![[INNERTYPEDEF:[0-9]+]], ![[UNNAMEDENUM:[0-9]+]], ![[INNERSTRUCT:[0-9]+]]}
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
