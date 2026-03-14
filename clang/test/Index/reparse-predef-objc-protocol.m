// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 3 local %s -I %S/Inputs
#import "declare-objc-predef.h"
// PR20633

// CHECK: declare-objc-predef.h:1:8: ObjCInterfaceDecl=Protocol:1:8 Extent=[1:1 - 1:16]
// CHECK: declare-objc-predef.h:1:8: ObjCClassRef=Protocol:1:8 Extent=[1:8 - 1:16]
// CHECK: declare-objc-predef.h:2:16: StructDecl=objc_class:2:16 Extent=[2:9 - 2:26]
// CHECK: declare-objc-predef.h:2:28: TypedefDecl=Class:2:28 (Definition) Extent=[2:1 - 2:33]
// CHECK: declare-objc-predef.h:2:16: TypeRef=struct objc_class:2:16 Extent=[2:16 - 2:26]
