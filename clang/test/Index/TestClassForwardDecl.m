// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-file-scan %t.ast %s | FileCheck -check-prefix=scan %s
// RUN: c-index-test -test-load-tu %t.ast local | FileCheck -check-prefix=load %s

// This test checks how the @class resolves as a cursor when the @interface is implicitly defined.
// See TestClassDecl.m for the corresponding test case. (<rdar://problem/7383421>)

@class Foo;

void function(Foo * arg)
{
    // nothing here.
}

// CHECK-scan: [1:1 - 8:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [8:1 - 8:8] ObjCInterfaceDecl=Foo:8:8
// CHECK-scan: [8:8 - 8:11] ObjCClassRef=Foo:8:8
// CHECK-scan: [8:11 - 10:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [10:1 - 10:15] FunctionDecl=function:10:6 (Definition)
// CHECK-scan: [10:15 - 10:18] ObjCClassRef=Foo:8:8
// CHECK-scan: [10:18 - 10:24] ParmDecl=arg:10:21 (Definition)
// CHECK-scan: [10:24 - 11:1] FunctionDecl=function:10:6 (Definition)
// CHECK: [11:1 - 13:2] CompundStmt=















// CHECK-load: TestClassForwardDecl.m:10:6: FunctionDecl=function:10:6 (Definition)
// CHECK-load: TestClassForwardDecl.m:10:21: ParmDecl=arg:10:21

