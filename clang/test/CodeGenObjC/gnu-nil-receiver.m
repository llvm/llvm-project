// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-arc -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s

typedef struct {
  int x[12];
} Big;

@protocol P
- (Big) foo;
- (Big) fooConsuming: (__attribute__((ns_consumed)) id) arg;
- (_Complex float) complex;
@end

@interface SuperClass
- (Big) foo;
@end

@implementation TestClass : SuperClass
//   Check that we don't do a nil check when messaging self in ARC
//   (which forbids reassigning self)
// CHECK-LABEL: define{{.*}} void @_i_TestClass__test_self_send(
// CHECK-NOT:   icmp
// CHECK:       @objc_msg_lookup_sender
- (void) test_self_send {
  Big big = [self foo];
}

//   Check that we don't do a nil test when messaging super.
// CHECK-LABEL: define{{.*}} void @_i_TestClass__test_super_send(
// CHECK-NOT:   icmp
// CHECK:       @objc_msg_lookup_super
- (void) test_super_send {
  Big big = [super foo];
}
@end

// CHECK-LABEL: define{{.*}} void @test_loop_zeroing(
// CHECK:         [[P:%.*]] = alloca ptr,
// CHECK:         [[BIG:%.*]] = alloca %struct.Big,
// CHECK:         br label %for.cond
//
// CHECK:       for.cond:
// CHECK-NEXT:    [[RECEIVER:%.*]] = load ptr, ptr [[P]],
// CHECK-NEXT:    [[ISNIL:%.*]] = icmp eq ptr [[RECEIVER]], null
// CHECK-NEXT:    br i1 [[ISNIL]], label %nilReceiverCleanup, label %msgSend
//
// CHECK:       msgSend:
// CHECK:         @objc_msg_lookup_sender
// CHECK:         call void {{%.*}}({{.*}} [[BIG]],
// CHECK:         br label %continue
//
// CHECK:       nilReceiverCleanup:
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[BIG]], i8 0, i64 48, i1 false)
// CHECK-NEXT:    br label %continue
//
// CHECK:       continue:
// CHECK-NEXT:    br label %for.cond
void test_loop_zeroing(id<P> p) {
  for (;;) {
    Big big = [p foo];
  }
}

// CHECK-LABEL: define{{.*}} void @test_zeroing_and_consume(
// CHECK:         [[P:%.*]] = alloca ptr,
// CHECK:         [[Q:%.*]] = alloca ptr,
// CHECK:         [[BIG:%.*]] = alloca %struct.Big,
// CHECK:         br label %for.cond
//
// CHECK:       for.cond:
// CHECK-NEXT:    [[RECEIVER:%.*]] = load ptr, ptr [[P]],
// CHECK-NEXT:    [[Q_LOAD:%.*]] = load ptr, ptr [[Q]],
// CHECK-NEXT:    [[Q_RETAIN:%.*]] = call ptr @llvm.objc.retain(ptr [[Q_LOAD]])
// CHECK-NEXT:    [[ISNIL:%.*]] = icmp eq ptr [[RECEIVER]], null
// CHECK-NEXT:    br i1 [[ISNIL]], label %nilReceiverCleanup, label %msgSend
//
// CHECK:       msgSend:
// CHECK:         @objc_msg_lookup_sender
// CHECK:         call void {{%.*}}({{.*}} [[BIG]],
// CHECK:         br label %continue
//
// CHECK:       nilReceiverCleanup:
// CHECK-NEXT:    call void @llvm.objc.release(ptr [[Q_RETAIN]])
// CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 [[BIG]], i8 0, i64 48, i1 false)
// CHECK-NEXT:    br label %continue
//
// CHECK:       continue:
// CHECK-NEXT:    br label %for.cond
void test_zeroing_and_consume(id<P> p, id q) {
  for (;;) {
    Big big = [p fooConsuming: q];
  }
}

// CHECK-LABEL: define{{.*}} void @test_complex(
// CHECK:         [[P:%.*]] = alloca ptr,
// CHECK:         [[RECEIVER:%.*]] = load ptr, ptr [[P]],
// CHECK-NEXT:    [[ISNIL:%.*]] = icmp eq ptr [[RECEIVER]], null
// CHECK-NEXT:    br i1 [[ISNIL]], label %continue, label %msgSend
// CHECK:       msgSend:
// CHECK:         @objc_msg_lookup_sender
// CHECK:         br label %continue
// CHECK:       continue:
// CHECK-NEXT:    phi float
// CHECK-NEXT:    phi float
void test_complex(id<P> p) {
  _Complex float f = [p complex];
}
