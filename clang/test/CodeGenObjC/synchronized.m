// RUN: %clang_cc1 -emit-llvm -triple i686-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -o - %s -O2 | FileCheck %s

@interface MyClass
{
}
- (void)method;
@end

@implementation MyClass

// CHECK: define internal void @"\01-[MyClass method]"
- (void)method
{
  // CHECK: call i32 @objc_sync_enter
  // CHECK: call void @objc_exception_try_enter
  // CHECK: call i32 @_setjmp
  @synchronized(self) {
  }
}

@end

// CHECK-LABEL: define{{.*}} void @foo(
void foo(id a) {
  // CHECK: [[A:%.*]] = alloca i8*
  // CHECK: [[SYNC:%.*]] = alloca i8*

  // CHECK:      store i8* [[AVAL:%.*]], i8** [[A]]
  // CHECK-NEXT: call i32 @objc_sync_enter(i8* [[AVAL]])
  // CHECK-NEXT: store i8* [[AVAL]], i8** [[SYNC]]
  // CHECK-NEXT: call void @objc_exception_try_enter
  // CHECK:      call i32 @_setjmp
  @synchronized(a) {
    // This is unreachable, but the optimizers can't know that.
    // CHECK: call void asm sideeffect "", "=*m,=*m,=*m"(i8** nonnull [[A]], i8** nonnull [[SYNC]]
    // CHECK: call i32 @objc_sync_exit
    // CHECK: call i8* @objc_exception_extract
    // CHECK: call void @objc_exception_throw
    // CHECK: unreachable

    // CHECK:      call void @objc_exception_try_exit
    // CHECK:      [[T:%.*]] = load i8*, i8** [[SYNC]]
    // CHECK-NEXT: call i32 @objc_sync_exit
    // CHECK: ret void
    return;
  }

}

// CHECK-LABEL: define{{.*}} i32 @f0(
int f0(id a) {
  // TODO: we can optimize the ret to a constant if we can figure out
  // either that x isn't stored to within the synchronized block or
  // that the synchronized block can't longjmp.

  // CHECK: [[X:%.*]] = alloca i32
  // CHECK: store i32 1, i32* [[X]]
  int x = 0;
  @synchronized((x++, a)) {    
  }

  // CHECK: [[T:%.*]] = load i32, i32* [[X]]
  // CHECK: ret i32 [[T]]
  return x;
}

// CHECK-LABEL: define{{.*}} void @f1(
void f1(id a) {
  // Check that the return doesn't go through the cleanup.
  extern void opaque(void);
  opaque();

  // CHECK: call void @opaque()
  // CHECK-NEXT: ret void

  @synchronized(({ return; }), a) {
    return;
  }
}
