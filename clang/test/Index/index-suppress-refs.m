
#include "index-suppress-refs.h"

#define TYPEDEF(x) typedef int x
TYPEDEF(MyInt);

MyInt gx;

@class I;

@interface I(cat)
-(I*)meth;
@end

@class I;

@interface S : B<P>
-(void)meth:(B*)b :(id<P>)p;
@end

// RUN: env CINDEXTEST_SUPPRESSREFS=1 c-index-test -index-file %s | FileCheck %s
// CHECK:      [indexDeclaration]: kind: objc-class | name: I
// CHECK-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-NEXT: [indexDeclaration]: kind: objc-class | name: B
// CHECK-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-NEXT: [indexDeclaration]: kind: objc-protocol | name: P
// CHECK-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-NEXT: [indexDeclaration]: kind: typedef | name: MyInt
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: gx
// CHECK-NEXT: [indexDeclaration]: kind: objc-class | name: I
// CHECK-NEXT:      <ObjCContainerInfo>: kind: forward-ref
// CHECK-NEXT: [indexDeclaration]: kind: objc-category | name: cat
// CHECK-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-NEXT:      <ObjCCategoryInfo>: class: kind: objc-class | name: I
// CHECK-NEXT: [indexDeclaration]: kind: objc-instance-method | name: meth
// CHECK-NOT:  [indexEntityReference]: kind: objc-class | name: I
// CHECK-NOT:  [indexDeclaration]: kind: objc-class | name: I
// CHECK-NEXT: [indexDeclaration]: kind: objc-class | name: S
// CHECK-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-NEXT:      <base>: kind: objc-class | name: B
// CHECK-NEXT:      <protocol>: kind: objc-protocol | name: P
// CHECK-NEXT: [indexDeclaration]: kind: objc-instance-method | name: meth::
// CHECK-NOT:  [indexEntityReference]: kind: objc-class | name: B
// CHECK-NOT:  [indexEntityReference]: kind: objc-protocol | name: P