/* Run lines are at the end, since line/column matter in this test. */

@interface A
- (void)method:(int)x;
@end

@implementation A
- (void)method:(int)x {
  A *a = [A method:1];
  blarg * blah = wibble;
  A *a2;
  z = [a2 method:1];
  blah ? blech : [a2 method:1];
  (a * a2)([a2 method:1]);
  B *a = [a2 method:1];
}
@end

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:9:20 %s 2>%t | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: not grep error %t
// CHECK-CC1: NotImplemented:{ResultType char[]}{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC1-NOT: NotImplemented:{TypedText _Bool}
// CHECK-CC1: VarDecl:{ResultType A *}{TypedText a}
// CHECK-CC1: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )}

// Test case for fix committed in r145441.
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:9:20 %s -fms-compatibility | FileCheck -check-prefix=CHECK-CC1 %s

// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:10:25 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{ResultType char[]}{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC2: NotImplemented:{TypedText _Bool}
// CHECK-CC2: VarDecl:{ResultType A *}{TypedText a}
// CHECK-CC2: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )}
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:12:11 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType void}{TypedText method:}{Placeholder (int)} (32)
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:13:22 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:14:16 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:15:14 %s | FileCheck -check-prefix=CHECK-CC3 %s
