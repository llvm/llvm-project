// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck %s
// rdar://10849570

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

// CHECK: [[SIXTEEN:%.*]]  = call i8* @llvm.objc.loadWeakRetained(i8** {{%.*}})
// CHECK-NEXT:  [[SEVENTEEN:%.*]] = bitcast i8* [[SIXTEEN]] to {{%.*}}
// CHECK-NEXT:  [[NINETEEN:%.*]] = bitcast %0* [[SEVENTEEN]] to i8*
// CHECK-NEXT:  [[EIGHTEEN:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.6
// CHECK-NEXT:  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT:  [[TWENTY:%.*]] = bitcast %0* [[SEVENTEEN]] to i8*
// CHECK-NEXT:  call void @llvm.objc.release(i8* [[TWENTY]])

void test1(int cond) {
  extern void test34_sink(id *);
  __weak id weak;
  test34_sink(cond ? &weak : 0);
}

// CHECK-LABEL: define{{.*}} void @test1(
// CHECK: [[CONDADDR:%.*]] = alloca i32
// CHECK-NEXT: [[WEAK:%.*]] = alloca i8*
// CHECK-NEXT: [[INCRTEMP:%.*]] = alloca i8*
// CHECK-NEXT: [[CONDCLEANUPSAVE:%.*]] = alloca i8*
// CHECK-NEXT: [[CONDCLEANUP:%.*]] = alloca i1
// CHECK-NEXT: store i32
// CHECK-NEXT: store i8* null, i8** [[WEAK]]
// CHECK:  [[COND1:%.*]] = phi i8**
// CHECK-NEXT: [[ICRISNULL:%.*]] = icmp eq i8** [[COND1]], null
// CHECK-NEXT: [[ICRARGUMENT:%.*]] = select i1 [[ICRISNULL]], i8** null, i8** [[INCRTEMP]]
// CHECK-NEXT: store i1 false, i1* [[CONDCLEANUP]]
// CHECK-NEXT: br i1 [[ICRISNULL]], label [[ICRCONT:%.*]], label [[ICRCOPY:%.*]]
// CHECK:  [[ONE:%.*]] = call i8* @llvm.objc.loadWeakRetained(
// CHECK-NEXT: store i8* [[ONE]], i8** [[CONDCLEANUPSAVE]]
// CHECK-NEXT: store i1 true, i1* [[CONDCLEANUP]]
// CHECK-NEXT: store i8* [[ONE]], i8** [[INCRTEMP]]
// CHECK-NEXT: br label

// CHECK: call void @test34_sink(
// CHECK-NEXT: [[ICRISNULL1:%.*]] = icmp eq i8** [[COND1]], null
// CHECK-NEXT: br i1 [[ICRISNULL1]], label [[ICRDONE:%.*]], label [[ICRWRITEBACK:%.*]]
// CHECK:  [[TWO:%.*]] = load i8*, i8** [[INCRTEMP]]
// CHECK-NEXT:  [[THREE:%.*]] = call i8* @llvm.objc.storeWeak(
// CHECK-NEXT:  br label [[ICRDONE]]
// CHECK:  [[CLEANUPISACTIVE:%.*]] = load i1, i1* [[CONDCLEANUP]]
// CHECK-NEXT:  br i1 [[CLEANUPISACTIVE]], label [[CLEASNUPACTION:%.*]], label [[CLEANUPDONE:%.*]]

// CHECK: [[FOUR:%.*]] = load i8*, i8** [[CONDCLEANUPSAVE]]
// CHECK-NEXT: call void @llvm.objc.release(i8* [[FOUR]])
// CHECK-NEXT:  br label
// CHECK:  call void @llvm.objc.destroyWeak(i8** [[WEAK]])
// CHECK-NEXT: ret void
