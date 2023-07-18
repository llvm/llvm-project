// RUN: c-index-test -write-pch %t.ast -fobjc-nonfragile-abi -fblocks -x objective-c %s 
// RUN: c-index-test -test-file-scan %t.ast %s | FileCheck -check-prefix=CHECK-scan %s
// RUN: c-index-test -test-load-tu %t.ast local | FileCheck -check-prefix=CHECK-load %s

// This test checks how the @class resolves as a cursor when there is a real definition
// that follows. <rdar://problem/7383421>

@class Foo;

@interface Foo
@end

void function(Foo * arg)
{
    // nothing here.
}

// CHECK-scan: [1:1 - 8:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [8:1 - 8:8] ObjCInterfaceDecl=Foo:8:8
// CHECK-scan: [8:8 - 8:11] ObjCClassRef=Foo:10:12
// CHECK-scan: [8:11 - 10:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [10:1 - 11:5] ObjCInterfaceDecl=Foo:10:12
// CHECK-scan: [11:5 - 13:1] Invalid Cursor => NoDeclFound
// CHECK-scan: [13:1 - 13:15] FunctionDecl=function:13:6 (Definition)
// CHECK-scan: [13:15 - 13:18] ObjCClassRef=Foo:10:12
// CHECK-scan: [13:18 - 13:24] ParmDecl=arg:13:21 (Definition)
// CHECK-scan: [13:24 - 14:1] FunctionDecl=function:13:6 (Definition)
// CHECK-scan: [14:1 - 16:2] CompoundStmt=

// CHECK-load: TestClassDecl.m:10:12: ObjCInterfaceDecl=Foo:10:12 Extent=[10:1 - 11:5]
// CHECK-load: TestClassDecl.m:13:6: FunctionDecl=function:13:6 (Definition) Extent=[13:1 - 16:2]
// CHECK-load: TestClassDecl.m:13:21: ParmDecl=arg:13:21 (Definition) Extent=[13:15 - 13:24]

