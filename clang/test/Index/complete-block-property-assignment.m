// Note: the run lines follow their respective tests, since line/column
// matter in this test.

// rdar://28481726

void func(int x);
typedef int Foo;
typedef void (^FooBlock)(Foo *someParameter);

@interface Obj
@property (readwrite, nonatomic, copy) void (^onAction)(Obj *object);
@property (readwrite, nonatomic) int foo;
@end

@interface Test : Obj
@property (readwrite, nonatomic, copy) FooBlock onEventHandler;
@property (readonly, nonatomic, copy) void (^onReadonly)(int *someParameter);
@property (readonly, nonatomic, strong) Obj *obj;
@end

@implementation Test

#define SELFY self

- (void)test {
  self.foo = 2;
  [self takeInt: 2]; self.foo = 2;
  /* Comment */ self.foo = 2;
  SELFY.foo = 2
}

// RUN: c-index-test -code-completion-at=%s:26:8 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:27:27 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:28:22 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:29:9 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText foo} (35)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Obj *}{TypedText obj} (35)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText onAction}{LeftParen (}{Placeholder Obj *object}{RightParen )} (35)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void (^)(Obj *)}{TypedText onAction}{Equal  = }{Placeholder ^(Obj *object)} (38)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText onEventHandler}{LeftParen (}{Placeholder Foo *someParameter}{RightParen )} (35)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType FooBlock}{TypedText onEventHandler}{Equal  = }{Placeholder ^(Foo *someParameter)} (38)
// CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText onReadonly}{LeftParen (}{Placeholder int *someParameter}{RightParen )} (35)

- (void) takeInt:(int)x { }

- (int) testFailures {
  (self.foo);
  int x = self.foo;
  [self takeInt: self.foo];
  if (self.foo) {
    func(self.foo);
  }
  return self.foo;
}

// RUN: c-index-test -code-completion-at=%s:47:9 %s | FileCheck -check-prefix=CHECK-NO %s
// RUN: c-index-test -code-completion-at=%s:48:16 %s | FileCheck -check-prefix=CHECK-NO %s
// RUN: c-index-test -code-completion-at=%s:49:23 %s | FileCheck -check-prefix=CHECK-NO %s
// RUN: c-index-test -code-completion-at=%s:50:12 %s | FileCheck -check-prefix=CHECK-NO %s
// RUN: c-index-test -code-completion-at=%s:51:15 %s | FileCheck -check-prefix=CHECK-NO %s
// RUN: c-index-test -code-completion-at=%s:53:15 %s | FileCheck -check-prefix=CHECK-NO %s
// CHECK-NO: ObjCPropertyDecl:{ResultType int}{TypedText foo} (35)
// CHECK-NO-NEXT: ObjCPropertyDecl:{ResultType Obj *}{TypedText obj} (35)
// CHECK-NO-NEXT: ObjCPropertyDecl:{ResultType void (^)(Obj *)}{TypedText onAction} (35)
// CHECK-NO-NEXT: ObjCPropertyDecl:{ResultType FooBlock}{TypedText onEventHandler} (35)
// CHECK-NO-NEXT: ObjCPropertyDecl:{ResultType void (^)(int *)}{TypedText onReadonly} (35)

@end
