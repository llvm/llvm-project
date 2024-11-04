// RUN: env CLANG_COMPILER_OBJC_MESSAGE_TRACE_PATH=%t.txt %clang_cc1 -fsyntax-only -triple arm64-apple-macosx15.0.0 -I %S/Inputs %s
// RUN: cat %t.txt | FileCheck %s

// CHECK: {
// CHECK-NEXT:    "format-version": 1,
// CHECK-NEXT:    "target": "arm64-apple-macosx15.0.0",
// CHECK-NEXT:    "references": [
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "A",
// CHECK-NEXT:            "instance_method": "-[A m0]",
// CHECK-NEXT:            "declared_at": "[[HEADER_FILE:.*]]:2:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "B",
// CHECK-NEXT:            "instance_method": "-[B m0]",
// CHECK-NEXT:            "declared_at": "[[HEADER_FILE]]:6:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "B",
// CHECK-NEXT:            "class_method": "+[B m1]",
// CHECK-NEXT:            "declared_at": "[[HEADER_FILE]]:7:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "B",
// CHECK-NEXT:            "category_type": "Cat1",
// CHECK-NEXT:            "instance_method": "-[B(Cat1) m2]",
// CHECK-NEXT:            "declared_at": "[[SOURCE_FILE:.*]]:12:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "C",
// CHECK-NEXT:            "instance_method": "-[C m4]",
// CHECK-NEXT:            "declared_at": "[[HEADER_FILE]]:15:1 <Spelling=[[HEADER_FILE]]:11:14>",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "protocol_type": "P0",
// CHECK-NEXT:            "instance_method": "-[P0 m5]",
// CHECK-NEXT:            "declared_at": "[[HEADER_FILE]]:18:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "B",
// CHECK-NEXT:            "category_type": "",
// CHECK-NEXT:            "instance_method": "-[B() m6]",
// CHECK-NEXT:            "declared_at": "[[SOURCE_FILE]]:16:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "implementation_type": "B",
// CHECK-NEXT:            "instance_method": "-[B m7:arg1:]",
// CHECK-NEXT:            "declared_at": "[[SOURCE_FILE]]:20:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:            "interface_type": "B",
// CHECK-NEXT:            "category_implementation_type": "Cat1",
// CHECK-NEXT:            "instance_method": "-[B(Cat1) m8]",
// CHECK-NEXT:            "declared_at": "[[SOURCE_FILE]]:25:1",
// CHECK-NEXT:            "referenced_at_file_id": 1
// CHECK-NEXT:        }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "fileMap": [
// CHECK-NEXT:        {
// CHECK-NEXT:            "file_id": 1,
// CHECK-NEXT:            "file_path": "[[SOURCE_FILE]]"
// CHECK-NEXT:        }
// CHECK-NEXT:    ]
// CHECK-NEXT: }

#include "objc-method-tracing.h"

@interface B(Cat1)
-(void)m2;
@end

@interface B()
-(void)m6;
@end

@implementation B
-(void)m7:(int)i arg1:(float)f {
}
@end

@implementation B(Cat1)
-(void)m8 {
}
@end

void test0(A *a) {
  [a m0];
}

void test1(B *b) {
  [b m0];
}

void test2(B *b) {
  [B m1];
}

void test3(B *b) {
  [b m2];
}

void test4(C *c) {
  [c m4];
}

void test5(id<P0> p0) {
  [p0 m5];
}

void test6(B *b) {
  [b m6];
}

void test7(B *b) {
  [b m7:123 arg1:4.5f];
}

void test8(B *b) {
  [b m8];
}
