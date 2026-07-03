// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -I%p %s -o - | FileCheck %s
// Test that the location of the typedef points to the header file.
#line 1 "a.c"
#line 2 "b.h"
typedef int MyType;
#line 2 "a.c"

MyType a;

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "MyType", file: ![[HEADER:[0-9]+]], line: 2,
// CHECK: ![[HEADER]] = !DIFile(filename: "b.h",
