// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++14 -fblocks -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s
struct test
{
    int func() { return 1; }
    int (^block)() = ^{ return func(); };
};

int main(int argc, const char * argv[]) {
    test t;
    return t.block();
}

// CHECK: ![[TESTCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "test"
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_1",
// CHECK-SAME:             elements: ![[ELEMS:.*]])
// CHECK: ![[ELEMS]] = !{{{.*}}, ![[THIS:[0-9]+]]}
// CHECK: ![[THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this",
// CHECK-SAME:                       baseType: ![[TESTCT]],


