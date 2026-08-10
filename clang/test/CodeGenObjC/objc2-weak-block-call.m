// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP64 %s
// RUN: %clang_cc1 -fblocks -fobjc-gc -triple i386-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | FileCheck -check-prefix CHECK-LP32 %s

@interface MyView
- (void)MyView_sharedInit;
@end

void foo(MyView *(^obj)(void)) ;

@implementation MyView
- (void)MyView_sharedInit {
    
    __block __weak MyView *weakSelf = self;
    foo(
    ^{
	return weakSelf;
    });

}
@end

// CHECK-LP64: call ptr @objc_read_weak
// CHECK-LP32: call ptr @objc_read_weak

