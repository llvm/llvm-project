// RUN: %clang_cc1 -triple x86_64-pc-win32 -debug-info-kind=limited -gcodeview %s -emit-llvm -o - | FileCheck %s

typedef struct {
} test1;

test1 gv1;
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "test1"

struct {
} test2;
void *use_test2 = &test2;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "<unnamed-type-test2>"

typedef struct {
} *test3;
test3 gv3;
void *use_test3 = &gv3;

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "<unnamed-type-test3>"
