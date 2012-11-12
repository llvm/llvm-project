// The line and column layout of this test is significant. Run lines
// are at the end.
typedef void (^block_t)(float f, double d);
void f(int (^block)(int x, int y));
void g(block_t b);

void test_f() {

}

@interface A
- method:(int (^)(int x, int y))b;
- method2:(block_t)b;
@end

void test_A(A *a) {
  [a method:0];
}

@interface B
- method3:(int (^)(void))b;
@end

void test_B(B *b) {
  [b method3:^int(void){ return 0; }];
}

@interface C
- method4:(void(^)(void))arg;
- method5:(void(^)())arg5;
@end

void test_C(C *c) {
  [c method4:^{}];
}

@interface D
- method6:(void(^)(block_t block))arg;
@end

void test_D(D *d) {
  [d method6:0];
}

// RUN: c-index-test -code-completion-at=%s:8:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText f}{LeftParen (}{Placeholder ^int(int x, int y)block}{RightParen )} (50)
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText g}{LeftParen (}{Placeholder ^(float f, double d)b}{RightParen )} (50)
// RUN: c-index-test -code-completion-at=%s:17:6 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText method2:}{Placeholder ^(float f, double d)b} (35)
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText method:}{Placeholder ^int(int x, int y)b} (35)
// RUN: c-index-test -code-completion-at=%s:25:6 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType id}{TypedText method3:}{Placeholder ^int(void)b} (35)
// RUN: c-index-test -code-completion-at=%s:34:6 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType id}{TypedText method4:}{Placeholder ^(void)arg} (35)
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType id}{TypedText method5:}{Placeholder ^(void)arg5} (35)
// RUN: c-index-test -code-completion-at=%s:25:15 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: TypedefDecl:{TypedText block_t} (50)
// CHECK-CC5: TypedefDecl:{TypedText Class} (50)
// CHECK-CC5-NOT: test_A
// CHECK-CC5: {TypedText union} (50)

// RUN: c-index-test -code-completion-at=%s:42:6 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{ResultType id}{TypedText method6:}{Placeholder ^(block_t block)arg} (35)

