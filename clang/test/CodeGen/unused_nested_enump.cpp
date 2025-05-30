// RUN: %clang_cc1 -debug-info-kind=unused-types  -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm -o - %s | FileCheck --check-prefix=NOUNUSEDTYPE %s

struct Type {
    enum { Unused };
    int value = 0;
};
int main() {
    Type t;
    return t.value;
}

// CHECK:  !DICompositeType(tag: DW_TAG_enumeration_type
// CHECK-SAME: scope: ![[STRUCT:[0-9]+]]
// CHECK-SAME: elements: ![[ELEMENTS:[0-9]+]]

// CHECK: ![[STRUCT]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Type"

// CHECK: ![[ELEMENTS]] = !{![[ENUMERATOR:[0-9]+]]}
// CHECK: ![[ENUMERATOR]] = !DIEnumerator(name: "Unused", value: 0


// NOUNUSEDTYPE-NOT: !DIEnumerator(name: "Unused"
