// Note: the run lines follow their respective tests, since line/column
// matter in this test.

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

// RUN: c-index-test -code-completion-at=%s:8:22 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:9:38 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:9:29 %s | FileCheck -check-prefix=CHECK-NONE %s
// RUN: c-index-test -code-completion-at=%s:9:34 %s | FileCheck -check-prefix=CHECK-NONE %s
// RUN: c-index-test -code-completion-at=%s:10:34 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:10:46 %s | FileCheck %s

// RUN: c-index-test -code-completion-at=%s:16:22 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:19:38 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:22:34 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:22:46 %s | FileCheck %s

// CHECK: macro definition:{TypedText NS_DESIGNATED_INITIALIZER} (70)

// CHECK-NONE-NOT: NS_DESIGNATED_INITIALIZER
