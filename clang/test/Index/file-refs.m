@class Foo;

@interface Foo
-(id)setWithInt:(int)i andFloat:(float)f;
@end

@implementation Foo
-(id)setWithInt:(int)i andFloat:(float)f {
  return self;
}
@end

void test(Foo *foo) {
  [foo setWithInt:0 andFloat:0];
  [foo setWithInt: 2 andFloat: 3];
}

@protocol Prot1
-(void)protMeth;
@end

@protocol Prot2<Prot1>
@end

@interface Base<Prot2>
@end

@interface Sub : Base
-(void)protMeth;
@end

@implementation Sub
-(void)protMeth {}
@end

void test2(Sub *s, id<Prot1> p) {
  [s protMeth];
  [p protMeth];
}


// RUN: c-index-test \

// RUN:  -file-refs-at=%s:7:18 \
// CHECK:      ObjCImplementationDecl=Foo:7:17 (Definition)

// FIXME: There should not be 2 for the same range.
// CHECK-NEXT: ObjCInterfaceDecl=Foo:1:8 =[1:8 - 1:11]
// CHECK-NEXT: ObjCClassRef=Foo:3:12 =[1:8 - 1:11]

// CHECK-NEXT: ObjCInterfaceDecl=Foo:3:12 =[3:12 - 3:15]
// CHECK-NEXT: ObjCImplementationDecl=Foo:7:17 (Definition) =[7:17 - 7:20]
// CHECK-NEXT: ObjCClassRef=Foo:3:12 =[13:11 - 13:14]

// RUN:  -file-refs-at=%s:4:10 \
// CHECK-NEXT: ObjCInstanceMethodDecl=setWithInt:andFloat::4:6
// CHECK-NEXT: ObjCInstanceMethodDecl=setWithInt:andFloat::4:6 =[4:6 - 4:16]
// CHECK-NEXT: ObjCInstanceMethodDecl=setWithInt:andFloat::8:6 (Definition) =[8:6 - 8:16]
// CHECK-NEXT: ObjCMessageExpr=setWithInt:andFloat::4:6 =[14:8 - 14:18]
// CHECK-NEXT: ObjCMessageExpr=setWithInt:andFloat::4:6 =[15:8 - 15:18]

// RUN:  -file-refs-at=%s:15:27 \
// CHECK-NEXT: ObjCMessageExpr=setWithInt:andFloat::4:6
// CHECK-NEXT: ObjCInstanceMethodDecl=setWithInt:andFloat::4:6 =[4:24 - 4:32]
// CHECK-NEXT: ObjCInstanceMethodDecl=setWithInt:andFloat::8:6 (Definition) =[8:24 - 8:32]
// CHECK-NEXT: ObjCMessageExpr=setWithInt:andFloat::4:6 =[14:21 - 14:29]
// CHECK-NEXT: ObjCMessageExpr=setWithInt:andFloat::4:6 =[15:22 - 15:30]

// RUN:  -file-refs-at=%s:18:13 \
// CHECK-NEXT: ObjCProtocolDecl=Prot1:18:11 (Definition)
// CHECK-NEXT: ObjCProtocolDecl=Prot1:18:11 (Definition) =[18:11 - 18:16]
// CHECK-NEXT: ObjCProtocolRef=Prot1:18:11 =[22:17 - 22:22]
// CHECK-NEXT: ObjCProtocolRef=Prot1:18:11 =[36:23 - 36:28]

// RUN:  -file-refs-at=%s:38:10 \
// CHECK-NEXT: ObjCMessageExpr=protMeth:19:8
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:19:8 =[19:8 - 19:16]
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:29:8 [Overrides @19:8] =[29:8 - 29:16]
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:33:8 (Definition) [Overrides @19:8] =[33:8 - 33:16]
// CHECK-NEXT: ObjCMessageExpr=protMeth:29:8 =[37:6 - 37:14]
// CHECK-NEXT: ObjCMessageExpr=protMeth:19:8 =[38:6 - 38:14]

// RUN:  -file-refs-at=%s:33:12 \
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:33:8 (Definition) [Overrides @19:8]
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:19:8 =[19:8 - 19:16]
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:29:8 [Overrides @19:8] =[29:8 - 29:16]
// CHECK-NEXT: ObjCInstanceMethodDecl=protMeth:33:8 (Definition) [Overrides @19:8] =[33:8 - 33:16]
// CHECK-NEXT: ObjCMessageExpr=protMeth:29:8 =[37:6 - 37:14]
// CHECK-NEXT: ObjCMessageExpr=protMeth:19:8 =[38:6 - 38:14]

// RUN:   %s | FileCheck %s
