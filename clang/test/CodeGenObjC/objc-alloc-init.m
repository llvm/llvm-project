// RUN: %clang_cc1 %s -fobjc-exceptions -fexceptions -fobjc-runtime=macosx-10.14.4 -emit-llvm -O0 -o - | FileCheck %s --check-prefix=OPTIMIZED --check-prefix=EITHER
// RUN: %clang_cc1 %s -fobjc-exceptions -fexceptions -fobjc-runtime=macosx-10.14.3 -emit-llvm -O0 -o - | FileCheck %s --check-prefix=NOT_OPTIMIZED --check-prefix=EITHER
// RUN: %clang_cc1 %s -fobjc-exceptions -fexceptions -fobjc-runtime=ios-12.2 -emit-llvm -O0 -o - | FileCheck %s --check-prefix=OPTIMIZED --check-prefix=EITHER
// RUN: %clang_cc1 %s -fobjc-exceptions -fexceptions -fobjc-runtime=ios-12.1 -emit-llvm -O0 -o - | FileCheck %s --check-prefix=NOT_OPTIMIZED --check-prefix=EITHER

@interface X
+(X *)alloc;
-(X *)init;
@end

void f() {
  [[X alloc] init];
  // OPTIMIZED: call i8* @objc_alloc_init(
  // NOT_OPTIMIZED: call i8* @objc_alloc(

  @try {
    [[X alloc] init];
  } @catch (X *x) {
  }
  // OPTIMIZED: invoke i8* @objc_alloc_init(
  // NOT_OPTIMIZED: invoke i8* @objc_alloc(
}

@interface Y : X
+(void)meth;
-(void)instanceMeth;
@end

@implementation Y
+(void)meth {
  [[self alloc] init];
  // OPTIMIZED: call i8* @objc_alloc_init(
  // NOT_OPTIMIZED: call i8* @objc_alloc(
}
-(void)instanceMeth {
  // EITHER-NOT: call i8* @objc_alloc
  // EITHER: call {{.*}} @objc_msgSend
  // EITHER: call {{.*}} @objc_msgSend
  [[self alloc] init];
}
@end

// rdar://48247290
@interface Base
-(instancetype)init;
@end

@interface Derived : Base
@end
@implementation Derived
-(void)meth {
  [super init];
}
@end
