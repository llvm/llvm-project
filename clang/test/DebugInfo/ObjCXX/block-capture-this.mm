// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++14 -fblocks -debug-info-kind=standalone -emit-llvm %s -o - | FileCheck %s
struct test
{
    int func() { return 1; }
    int (^block)() = ^{ return func(); };
    int viaLambda = [this] { return ^{ return func(); }(); }();
};

int main(int argc, const char * argv[]) {
    test t;
    return t.block();
}

// The initializers are also emitted here, outside any member function.
void f() {
    test t{};
}

// CHECK: ![[TESTCT:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "test"
// CHECK: ![[THISPTR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[TESTCT]], size: 64)

// The line-5 block gets a literal type per emitting function (the constructor
// and f), sharing the member list; the line-6 block is inside the lambda.
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_{{[0-9]+}}", {{.*}} line: 5, size: 320, elements: ![[ELEMS5:[0-9]+]])
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_{{[0-9]+}}", {{.*}} line: 5, size: 320, elements: ![[ELEMS5]])
// CHECK-DAG: ![[ELEMS5]] = !{{{.*}}, ![[THIS5:[0-9]+]]}
// CHECK-DAG: ![[THIS5]] = !DIDerivedType(tag: DW_TAG_member, name: "this", {{.*}} line: 5, baseType: ![[THISPTR]], size: 64, offset: 256, flags: DIFlagPublic)

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_literal_{{[0-9]+}}", {{.*}} line: 6, size: 320, elements: ![[ELEMS6:[0-9]+]])
// CHECK-DAG: ![[ELEMS6]] = !{{{.*}}, ![[THIS6:[0-9]+]]}
// CHECK-DAG: ![[THIS6]] = !DIDerivedType(tag: DW_TAG_member, name: "this", {{.*}} line: 6, baseType: ![[THISPTR]], size: 64, offset: 256, flags: DIFlagPublic)
