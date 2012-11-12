// Note: the run lines follow their respective tests, since line/column
// matter in this test.

typedef int Bool;

@interface A
- (void)add:(int)x to:(int)y;
+ (void)select:(Bool)condition first:(int)x second:(int)y;
- (void)last;
+ (void)last;
@end

@interface B : A
- (void)add:(int)x to:(int)y;
+ (void)select:(Bool)condition first:(int)x second:(int)y;
@end

@implementation B
- (void)add:(int)a to:(int)b {
  [super add:a to:b];
}

+ (void)select:(Bool)condition first:(int)a second:(int)b {
  [super selector:condition first:a second:b];
   super selector:condition first:a second:b];
}
@end

@interface A (Cat)
- (void)multiply:(int)x by:(int)y;
@end

@interface C : A
@end

@implementation C
- (void)multiply:(int)a by:(int)b {
  [super multiply:a by:b];
}
@end

// Check "super" completion as a message receiver.
// RUN: c-index-test -code-completion-at=%s:20:4 %s | FileCheck -check-prefix=CHECK-ADD-RECEIVER %s
// CHECK-ADD-RECEIVER: ObjCInstanceMethodDecl:{ResultType void}{TypedText super}{HorizontalSpace  }{Text add:}{Placeholder a}{HorizontalSpace  }{Text to:}{Placeholder b} (20)

// RUN: c-index-test -code-completion-at=%s:24:4 %s | FileCheck -check-prefix=CHECK-SELECT-RECEIVER %s
// CHECK-SELECT-RECEIVER: ObjCClassMethodDecl:{ResultType void}{TypedText super}{HorizontalSpace  }{Text select:}{Placeholder condition}{HorizontalSpace  }{Text first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (20)

// Check "super" completion at the first identifier
// RUN: c-index-test -code-completion-at=%s:20:10 %s | FileCheck -check-prefix=CHECK-ADD-ADD %s
// CHECK-ADD-ADD: ObjCInstanceMethodDecl:{ResultType void}{TypedText add:}{Placeholder a}{HorizontalSpace  }{Text to:}{Placeholder b} (20)
// CHECK-ADD-ADD-NOT: add
// CHECK-ADD-ADD: ObjCInstanceMethodDecl:{ResultType void}{TypedText last} (35)

// RUN: c-index-test -code-completion-at=%s:24:10 %s | FileCheck -check-prefix=CHECK-SELECTOR-SELECTOR %s
// CHECK-SELECTOR-SELECTOR-NOT: x
// CHECK-SELECTOR-SELECTOR: ObjCClassMethodDecl:{ResultType void}{TypedText last} (35)
// CHECK-SELECTOR-SELECTOR: ObjCClassMethodDecl:{ResultType void}{TypedText select:}{Placeholder condition}{HorizontalSpace  }{Text first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (20)

// Check "super" completion at the second identifier
// RUN: c-index-test -code-completion-at=%s:20:16 %s | FileCheck -check-prefix=CHECK-ADD-TO %s
// CHECK-ADD-TO: ObjCInstanceMethodDecl:{ResultType void}{Informative add:}{TypedText to:}{Placeholder b} (20)

// RUN: c-index-test -code-completion-at=%s:24:28 %s | FileCheck -check-prefix=CHECK-SELECTOR-FIRST %s
// CHECK-SELECTOR-FIRST: ObjCClassMethodDecl:{ResultType void}{Informative select:}{TypedText first:}{Placeholder a}{HorizontalSpace  }{Text second:}{Placeholder b} (20)

// Check "super" completion at the third identifier
// RUN: c-index-test -code-completion-at=%s:24:37 %s | FileCheck -check-prefix=CHECK-SELECTOR-SECOND %s
// CHECK-SELECTOR-SECOND: ObjCClassMethodDecl:{ResultType void}{Informative select:}{Informative first:}{TypedText second:}{Placeholder b} (20)

// Check "super" completion with missing '['.
// RUN: c-index-test -code-completion-at=%s:25:10 %s | FileCheck -check-prefix=CHECK-SELECTOR-SELECTOR %s
// RUN: c-index-test -code-completion-at=%s:25:28 %s | FileCheck -check-prefix=CHECK-SELECTOR-FIRST %s
// RUN: c-index-test -code-completion-at=%s:25:37 %s | FileCheck -check-prefix=CHECK-SELECTOR-SECOND %s

// Check "super" completion for a method declared in a category.
// RUN: c-index-test -code-completion-at=%s:38:10 %s | FileCheck -check-prefix=CHECK-IN-CATEGORY %s
// CHECK-IN-CATEGORY: ObjCInstanceMethodDecl:{ResultType void}{TypedText add:}{Placeholder (int)}{HorizontalSpace  }{TypedText to:}{Placeholder (int)} (35)
// CHECK-IN-CATEGORY: ObjCInstanceMethodDecl:{ResultType void}{TypedText last} (35)
// CHECK-IN-CATEGORY: ObjCInstanceMethodDecl:{ResultType void}{TypedText multiply:}{Placeholder a}{HorizontalSpace  }{Text by:}{Placeholder b} (20)

