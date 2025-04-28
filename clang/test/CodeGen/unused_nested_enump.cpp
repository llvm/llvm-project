// RUN: %clang_cc1 -debug-info-kind=unused-types  -emit-llvm -o - %s | FileCheck %s

struct Type {
    enum { Unused };
    int value = 0;
};
int main() {
    Type t;
    return t.value;
}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type
// CHECK-SAME: scope: ![[STRUCT:[0-9]+]]
// CHECK-SAME: elements: ![[ELEMENTS:[0-9]+]]

// CHECK: ![[STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type
// CHECK-SAME: elements: ![[STRUCT_ELEMENTS:[0-9]+]]

// CHECK: ![[STRUCT_ELEMENTS]] = !{![[ENUM_MEMBER:[0-9]+]], ![[VALUE_MEMBER:[0-9]+]]}

// CHECK: ![[VALUE_MEMBER]] = !DIDerivedType(tag: DW_TAG_member
// CHECK-SAME: name: "value"
// CHECK-SAME: scope: ![[STRUCT]]

// CHECK: ![[ELEMENTS]] = !{![[ENUMERATOR:[0-9]+]]}
// CHECK: ![[ENUMERATOR]] = !DIEnumerator(name: "Unused", value: 0, isUnsigned: true)
