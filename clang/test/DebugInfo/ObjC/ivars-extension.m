// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s

// Make sure we generate debug symbols for ivars added by a class extension.

@interface I
{
    @public int a;
}
@end

void foo(I* pi) {
    // poking into pi for primary class ivars.
    int _a = pi->a;
}

@interface I()
{
    @public int b;
}
@end

void gorf (I* pg) {
    // poking into pg for ivars for class extension
    int _b = pg->b;
}

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "I"

// Check for "a".
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "a"
// CHECK-SAME:           line: 7
// CHECK-SAME:           baseType: ![[INT:[0-9]+]]
// CHECK-SAME:           size: 32
// CHECK-NOT:            offset:
// CHECK-SAME:           flags: DIFlagPublic
// CHECK: ![[INT]] = !DIBasicType(name: "int"

// Make sure we don't output the same type twice.
// CHECK-NOT: !DICompositeType(tag: DW_TAG_structure_type, name: "I"

// Check for "b".
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "b"
// CHECK-SAME:           line: 18
// CHECK-SAME:           baseType: ![[INT]]
// CHECK-SAME:           size: 32
// CHECK-NOT:            offset:
// CHECK-SAME:           flags: DIFlagPublic
