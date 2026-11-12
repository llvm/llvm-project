// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck %s

@interface NSObject @end

@interface SomeClass : NSObject
- (id) init;
@end

@implementation SomeClass
- (void)foo {
}
- (id) init {
    return 0;
}
+ alloc { return 0; }
@end

int main (int argc, const char * argv[]) {
    @autoreleasepool {
        SomeClass *objPtr1 = [[SomeClass alloc] init];
        __weak SomeClass *weakRef = objPtr1;

        [weakRef foo];

        objPtr1 = (void *)0;
        return 0;
    }
}

// CHECK: [[SIXTEEN:%.*]]  = call ptr @llvm.objc.loadWeakRetained(ptr {{%.*}})
// CHECK-NEXT:  [[EIGHTEEN:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.6
// CHECK-NEXT:  call void @objc_msgSend
// CHECK-NEXT:  call void @llvm.objc.release(ptr [[SIXTEEN]])

void test1(int cond) {
  extern void test34_sink(id *);
  __weak id weak;
  test34_sink(cond ? &weak : 0);
}

// CHECK-LABEL: define{{.*}} void @test1(
// CHECK: [[CONDADDR:%.*]] = alloca i32
// CHECK-NEXT: [[WEAK:%.*]] = alloca ptr
// CHECK-NEXT: [[INCRTEMP:%.*]] = alloca ptr
// CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca ptr
// CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
// CHECK-NEXT: store i32
// CHECK-NEXT: store ptr null, ptr [[WEAK]]
// CHECK:  [[COND1:%.*]] = phi ptr
// CHECK-NEXT: [[ICRISNULL:%.*]] = icmp eq ptr [[COND1]], null
// CHECK-NEXT: [[ICRARGUMENT:%.*]] = select i1 [[ICRISNULL]], ptr null, ptr [[INCRTEMP]]
// CHECK-NEXT: store i1 false, ptr [[CONDCLEANUP]]
// CHECK-NEXT: br i1 [[ICRISNULL]], label [[ICRCONT:%.*]], label [[ICRCOPY:%.*]]
// CHECK:  [[ONE:%.*]] = call ptr @llvm.objc.loadWeakRetained(
// CHECK-NEXT: store ptr [[ONE]], ptr [[CONDCLEANUPSAVE]]
// CHECK-NEXT: store i1 true, ptr [[CONDCLEANUP]]
// CHECK-NEXT: store ptr [[ONE]], ptr [[INCRTEMP]]
// CHECK-NEXT: br label

// CHECK: call void @test34_sink(
// CHECK-NEXT: [[ICRISNULL1:%.*]] = icmp eq ptr [[COND1]], null
// CHECK-NEXT: br i1 [[ICRISNULL1]], label [[ICRDONE:%.*]], label [[ICRWRITEBACK:%.*]]
// CHECK:  [[TWO:%.*]] = load ptr, ptr [[INCRTEMP]]
// CHECK-NEXT:  [[THREE:%.*]] = call ptr @llvm.objc.storeWeak(
// CHECK-NEXT:  br label [[ICRDONE]]
// CHECK:  [[CLEANUPISACTIVE:%.*]] = load i1, ptr [[CONDCLEANUP]]
// CHECK-NEXT:  br i1 [[CLEANUPISACTIVE]], label [[CLEASNUPACTION:%.*]], label [[CLEANUPDONE:%.*]]

// CHECK: [[FOUR:%.*]] = load ptr, ptr [[CONDCLEANUPSAVE]]
// CHECK-NEXT: call void @llvm.objc.release(ptr [[FOUR]])
// CHECK-NEXT:  br label
// CHECK:  call void @llvm.objc.destroyWeak(ptr [[WEAK]])
// CHECK-NEXT: ret void
