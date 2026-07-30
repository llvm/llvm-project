@protocol NSObject
@end

@interface NSObject
@end

@interface Test<T : id, U : NSObject *> : NSObject
{
@public
	U myVar;
}
-(U)getit:(T)val;
-(void)apply:(void(^)(T, U))block;
-(void)apply2:(void(^_Nonnull)(T, U))block;

@property (strong) T prop;
@end

@interface MyClsA : NSObject
@end
@interface MyClsB : NSObject
@end

void test1(Test<MyClsA*, MyClsB*> *obj) {
  [obj ];
  obj.;
  obj->;
}

void test2(Test *obj) {
  [obj ];
  obj.;
  obj->;
}

@implementation Test
-(id)getit:(id)val {}
@end

void test3() {
  Test<> t;
  NSObject<> n;
}

// RUN: c-index-test -code-completion-at=%s:25:8 %s | FileCheck -check-prefix=CHECK-CC0 %s
// CHECK-CC0: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply2:}{Placeholder ^(MyClsA *, MyClsB *)block} (35)
// CHECK-CC0: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply:}{Placeholder ^(MyClsA *, MyClsB *)block} (35)
// CHECK-CC0: ObjCInstanceMethodDecl:{ResultType MyClsB *}{TypedText getit:}{Placeholder (MyClsA *)} (35)

// RUN: c-index-test -code-completion-at=%s:26:7 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType MyClsA *}{TypedText prop} (35)

// RUN: c-index-test -code-completion-at=%s:27:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCIvarDecl:{ResultType MyClsB *}{TypedText myVar} (35)

// RUN: c-index-test -code-completion-at=%s:31:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply2:}{Placeholder ^(id, NSObject *)block} (35)
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply:}{Placeholder ^(id, NSObject *)block} (35)
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType __kindof NSObject *}{TypedText getit:}{Placeholder (id)} (35)

// RUN: c-index-test -code-completion-at=%s:32:7 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCPropertyDecl:{ResultType id}{TypedText prop} (35)

// RUN: c-index-test -code-completion-at=%s:33:8 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCIvarDecl:{ResultType __kindof NSObject *}{TypedText myVar} (35)

// RUN: c-index-test -code-completion-at=%s:37:2 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText apply2:}{LeftParen (}{Text void (^)(id, NSObject *)}{RightParen )}{Text block} (40)
// CHECK-CC6: ObjCInstanceMethodDecl:{LeftParen (}{Text void}{RightParen )}{TypedText apply:}{LeftParen (}{Text void (^)(id, NSObject *)}{RightParen )}{Text block} (40)
// CHECK-CC6: ObjCInstanceMethodDecl:{LeftParen (}{Text NSObject *}{RightParen )}{TypedText getit:}{LeftParen (}{Text id}{RightParen )}{Text val} (40)
// CHECK-CC6: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText prop} (40)

// RUN: c-index-test -code-completion-at=%s:41:8 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ObjCInterfaceDecl:{TypedText MyClsA}
// RUN: c-index-test -code-completion-at=%s:42:12 %s > %t.out
// RUN: FileCheck -input-file=%t.out -check-prefix=CHECK-CC8 %s
// RUN: FileCheck -input-file=%t.out -check-prefix=CHECK-CC9 %s
// CHECK-CC8: ObjCProtocolDecl:{TypedText NSObject}
// CHECK-CC9-NOT: ObjCInterfaceDecl:{TypedText MyClsA}
