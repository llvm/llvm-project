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
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Foo (^)(int, Foo, FooBlock)}{TypedText blocker}{Equal  = }{Placeholder ^Foo(int x, Foo y, FooBlock foo)} (32)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText foo} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText fooBlock}{LeftParen (}{Placeholder Foo *someParameter}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType Test *}{TypedText getObject}{LeftParen (}{Placeholder int index}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText performA}{LeftParen (}{RightParen )} (35)
//CHECK-CC1-NEXT: ObjCPropertyDecl:{ResultType int}{TypedText performB}{LeftParen (}{Placeholder int x}{Comma , }{Placeholder int y}{RightParen )} (35)

@end

// rdar://25224416

@interface NoQualifierParens

@property(copy) void (^blockProperty)(void);
@property BarBlock blockProperty2;

@end

void noQualifierParens(NoQualifierParens *f) {
  [f setBlockProperty: ^{}];
}

// RUN: c-index-test -code-completion-at=%s:65:6 %s | FileCheck -check-prefix=CHECK-CC2 %s
//CHECK-CC2: ObjCInstanceMethodDecl:{ResultType void (^)(void)}{TypedText blockProperty} (35)
//CHECK-CC2-NEXT: ObjCInstanceMethodDecl:{ResultType BarBlock}{TypedText blockProperty2} (35)
//CHECK-CC2-NEXT: ObjCInstanceMethodDecl:{ResultType void}{TypedText setBlockProperty2:}{Placeholder BarBlock blockProperty2} (35)
//CHECK-CC2-NEXT: ObjCInstanceMethodDecl:{ResultType void}{TypedText setBlockProperty:}{Placeholder void (^)(void)blockProperty} (35)

@interface ClassProperties

@property(class) void (^explicit)();
@property(class, readonly) void (^explicitReadonly)();

@end

void classBlockProperties() {
  ClassProperties.explicit;
}

// RUN: c-index-test -code-completion-at=%s:82:19 %s | FileCheck -check-prefix=CHECK-CC3 %s
//CHECK-CC3: ObjCPropertyDecl:{ResultType void}{TypedText explicit}{LeftParen (}{RightParen )} (35)
//CHECK-CC3-NEXT: ObjCPropertyDecl:{ResultType void (^)()}{TypedText explicit}{Equal  = }{Placeholder ^(void)} (38)
//CHECK-CC3-NEXT: ObjCPropertyDecl:{ResultType void}{TypedText explicitReadonly}{LeftParen (}{RightParen )} (35)
