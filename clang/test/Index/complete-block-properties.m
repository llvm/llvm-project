// Note: the run lines follow their respective tests, since line/column
// matter in this test.

// Block invocations should be presented when completing properties in
// standalone statements.
// rdar://28846196

typedef int Foo;
typedef void (^FooBlock)(Foo *someParameter);
typedef int (^BarBlock)(int *);

@interface Obj

@property (readwrite, nonatomic, copy) void (^block)();
@property (readonly, nonatomic, copy) int (^performA)();
@property (readonly, nonatomic, copy) int (^performB)(int x, int y);
@property (readwrite, nonatomic, copy) Foo (^blocker)(int x, Foo y, FooBlock foo);

@end


@interface Test : Obj

@property (readonly, nonatomic, copy) FooBlock fooBlock;
@property (readonly, nonatomic, copy) BarBlock barBlock;
@property (readonly, nonatomic, copy) Test * (^getObject)(int index);
@property (readwrite, nonatomic) int foo;

@end

@implementation Test

- (void)test {
  self.foo = 2;
  int x = self.performA(); self.foo = 2;
  self.getObject(0).foo = 2;
}

// RUN: c-index-test -code-completion-at=%s:34:8 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:35:33 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:36:21 %s | FileCheck -check-prefix=CHECK-CC1 %s
//CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText barBlock}{LeftParen (}{Placeholder int *}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText block}{LeftParen (}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void (^)()}{TypedText block}{Equal  = }{Placeholder ^(void)} (38)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Foo}{TypedText blocker}{LeftParen (}{Placeholder int x}{Comma , }{Placeholder Foo y}{Comma , }{Placeholder ^(Foo *someParameter)foo}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Foo (^)(int, Foo, FooBlock)}{TypedText blocker}{Equal  = }{Placeholder ^Foo(int x, Foo y, FooBlock foo)} (38)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText foo} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText fooBlock}{LeftParen (}{Placeholder Foo *someParameter}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Test *}{TypedText getObject}{LeftParen (}{Placeholder int index}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText performA}{LeftParen (}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText performB}{LeftParen (}{Placeholder int x}{Comma , }{Placeholder int y}{RightParen )} (35)

@end
