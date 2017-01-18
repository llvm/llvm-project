// Note: the run lines follow their respective tests, since line/column
// matter in this test.

// rdar://21014571

#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))

@interface DesignatedInitializerCompletion

- (instancetype)init ;
- (instancetype)initWithFoo:(int)foo ;
- (instancetype)initWithX:(int)x andY:(int)y ;

@end

@implementation DesignatedInitializerCompletion

- (instancetype)init {
}

- (instancetype)initWithFoo:(int)foo {
}

- (instancetype)initWithX:(int)x andY:(int)y {
}

@end

// RUN: c-index-test -code-completion-at=%s:10:22 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:11:38 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:11:29 %s | FileCheck -check-prefix=CHECK-NONE %s
// RUN: c-index-test -code-completion-at=%s:11:34 %s | FileCheck -check-prefix=CHECK-NONE %s
// RUN: c-index-test -code-completion-at=%s:12:34 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:12:46 %s | FileCheck %s

// RUN: c-index-test -code-completion-at=%s:18:22 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:21:38 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:24:34 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:24:46 %s | FileCheck %s

// CHECK: macro definition:{TypedText NS_DESIGNATED_INITIALIZER} (70)

// CHECK-NONE-NOT: NS_DESIGNATED_INITIALIZER
