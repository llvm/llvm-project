// REQUIRES: system-darwin

// RUN: mkdir -p %t

// RUN: %clang -fobjc-export-direct-methods -S -emit-llvm   \
// RUN:   -target arm64-apple-darwin -O0 %s -o - -fobjc-arc \
// RUN:   | FileCheck %s

// RUN: %clang -fobjc-export-direct-methods -S -emit-llvm  \
// RUN:   -target arm64-apple-darwin -O0 %s -o -           \
// RUN:   | FileCheck --check-prefix=NO-ARC %s

// RUN: %clang -fobjc-export-direct-methods     \
// RUN:   -target arm64-apple-darwin -fobjc-arc \
// RUN:   -O2 -framework Foundation %s -o %t/shape

// RUN: %t/shape 1 2 3 4 | FileCheck %s --check-prefix=CHECK-EXE

// NO-ARC-NOT: autoreleaseReturnValue
// NO-ARC-NOT: retainAutoreleasedReturnValue
// NO-ARC-NOT: asm sideeffect "mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue"

#import <Foundation/Foundation.h>
#include "math.h"

@interface Shape: NSObject
@property(direct, readonly) int x;
@property(direct, readonly) int y;
@property(direct) Shape* innerShape;
@property(class) int numInstances;
@property(direct) int instanceId;
- (void) dealloc;
- (instancetype)initWithX:(int)x Y:(int)y __attribute__((objc_direct));
- (instancetype)initDefault __attribute__((objc_direct));
- (double) distanceFrom: (Shape *) __attribute__((ns_consumed)) s __attribute__((objc_direct));
+ (Shape *) default __attribute__((objc_direct));
- (instancetype) clone __attribute__((objc_direct));
@end

@implementation Shape
@dynamic numInstances;
static int numInstances=0;

- (void) dealloc {
  printf("Dealloc %d\n", self.instanceId);
}
- (instancetype)initWithX:(int)x Y:(int)y  {
  if (self = [super init]) {
    _x = x;
    _y = y;
    _innerShape = nil;
    _instanceId = numInstances;
    printf("Alloc %d\n", _instanceId);
    numInstances++;
  }
  return self;
}

// Thunk function should not release anything.
// CHECK-LABEL: define dso_local ptr @"-<Shape initDefault>"
// CHECK-NOT: call void @llvm.objc.storeStrong
// CHECK-LABEL: }
- (instancetype)initDefault {
  return [self initWithX:0 Y:0];
}

// CHECK-LABEL: define dso_local ptr @"+<Shape default>"
// CHECK: [[SHAPE:%.*]] = call ptr @"-<Shape initDefault>"
// CHECK-NEXT: [[AUTORELEASE_SHAPE:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[SHAPE]])
// CHECK-NEXT: ret ptr [[AUTORELEASE_SHAPE]]
// CHECK-LABEL: }
+ (Shape*) default {
  return [[Shape alloc] initDefault];
}

// CHECK-LABEL: define {{.*}} @"-<Shape clone>_inner"
// CHECK: [[CALL_INIT:%.*]] = call ptr @"-<Shape initWithX:Y:>"
// CHECK-NEXT: [[AUTORELEASE_CLONE:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[CALL_INIT]])
// CHECK-NEXT: ret ptr [[AUTORELEASE_CLONE]]
// CHECK-LABEL: }

// CHECK-LABEL: define {{.*}} @"-<Shape clone>"
// CHECK: [[CALL_INNER:%.*]] = call ptr @"-<Shape clone>_inner"
// CHECK-NEXT: call void asm sideeffect "mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue", ""()
// CHECK-NEXT: [[RETAINED:%.*]] = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[CALL_INNER]])
// CHECK-NEXT: store ptr [[RETAINED]], ptr [[RETADDR:%.*]]
// CHECK: [[RET:%.*]] = load ptr, ptr [[RETADDR]]
// CHECK-NEXT: [[AUTORELEASE_RET:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[RET]])
// CHECK-NEXT: ret ptr [[AUTORELEASE_RET]]
// CHECK-LABEL: }
- (instancetype) clone {
  return [[Shape alloc] initWithX:self.x Y:self.y];
}

// InnerFn will release the value since it is "consumed".
// CHECK: define dso_local double @"-<Shape distanceFrom:>_inner"(ptr noundef nonnull %{{.*}}, ptr noundef [[S:%.*]]) #0 {
// CHECK: {{%.*}} = alloca ptr
// CHECK: [[S_ADDR:%.*]] = alloca ptr
// CHECK: store ptr [[S]], ptr [[S_ADDR]]
// CHECK: call void @llvm.objc.storeStrong(ptr [[S_ADDR]], ptr null)

// Thunk function should not release anything even with ns_consumed
// CHECK-LABEL: define dso_local double @"-<Shape distanceFrom:>"
// CHECK-NOT: call void @llvm.objc.storeStrong
// CHECK-LABEL: }
- (double) distanceFrom:(Shape *) __attribute__((ns_consumed)) s __attribute__((objc_direct)) {
  double dist = sqrt((s.x - self.x) * (s.x - self.x) + (s.y - self.y) * (s.y - self.y));
  return dist;
}
@end

// CHECK-LABEL: define i32 @main
int main(int argc, char** argv) { // argv = ["1", "2", "3", "4"]
@autoreleasepool {
  // CHECK-EXE: Alloc
  Shape* classDefault = [Shape default];
  // CHECK-EXE-NEXT: Alloc
  Shape* s = [[Shape alloc] initWithX:atoi(argv[0]) Y:atoi(argv[1])];
  // CHECK-EXE-NEXT: Alloc
  Shape* t = [[Shape alloc] initWithX:atoi(argv[2]) Y:atoi(argv[3])];
  // CHECK-EXE-NEXT: Alloc
  Shape* zero = [[Shape alloc] initDefault];
  // CHECK-EXE-NEXT: Alloc
  Shape* anotherDefault = [Shape default];

  // CHECK: [[CALL_CLONE:%.*]] = call ptr @"-<Shape clone>"
  // CHECK-NEXT: call void asm sideeffect "mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue", ""()
  // CHECK-NEXT: {{%.*}} = call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[CALL_CLONE]])

  // CHECK-EXE-NEXT: Alloc [[CLOND_ID:.*]]
  Shape* cloned = [s clone];

  Shape* null = nil;
  // CHECK-EXE: Dist: 2.82
  printf("Dist: %lf\n", [s distanceFrom:t]);
  // CHECK-EXE-NEXT: Dist: 3.60
  printf("Dist: %lf\n", [zero distanceFrom:t]);
  // CHECK-EXE-NEXT: Dist: 3.60
  printf("Dist: %lf\n", [classDefault distanceFrom:t]);
  // CHECK-EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [s distanceFrom:s]);
  // CHECK-EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [classDefault distanceFrom:anotherDefault]);
  // CHECK-EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [null distanceFrom:zero]);
  // CHECK-EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [s distanceFrom:cloned]);

  // Cloned object should be released as well
  // CHECK-EXE: Dealloc [[CLOND_ID]]
}
  return 0;
}
