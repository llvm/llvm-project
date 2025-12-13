// REQUIRES: system-darwin

// RUN: mkdir -p %t

// RUN: %clang -Xclang -fobjc-direct-precondition-thunk   \
// RUN:   -target arm64-apple-darwin -fobjc-arc \
// RUN:   -O2 -framework Foundation %s -o %t/shape

// RUN: %t/shape 1 2 3 4 | FileCheck %s --check-prefix=EXE

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

- (instancetype)initDefault {
  return [self initWithX:0 Y:0];
}

+ (Shape*) default {
  return [[Shape alloc] initDefault];
}


- (instancetype) clone {
  return [[Shape alloc] initWithX:self.x Y:self.y];
}

- (double) distanceFrom:(Shape *) __attribute__((ns_consumed)) s __attribute__((objc_direct)) {
  double dist = sqrt((s.x - self.x) * (s.x - self.x) + (s.y - self.y) * (s.y - self.y));
  return dist;
}
@end

int main(int argc, char** argv) { // argv = ["1", "2", "3", "4"]
@autoreleasepool {
  // EXE: Alloc
  Shape* classDefault = [Shape default];
  // EXE-NEXT: Alloc
  Shape* s = [[Shape alloc] initWithX:atoi(argv[0]) Y:atoi(argv[1])];
  // EXE-NEXT: Alloc
  Shape* t = [[Shape alloc] initWithX:atoi(argv[2]) Y:atoi(argv[3])];
  // EXE-NEXT: Alloc
  Shape* zero = [[Shape alloc] initDefault];
  // EXE-NEXT: Alloc
  Shape* anotherDefault = [Shape default];

  // EXE-NEXT: Alloc
  Shape* cloned = [s clone];

  Shape* null = nil;
  // EXE: Dist: 2.82
  printf("Dist: %lf\n", [s distanceFrom:t]);
  // EXE-NEXT: Dist: 3.60
  printf("Dist: %lf\n", [zero distanceFrom:t]);
  // EXE-NEXT: Dist: 3.60
  printf("Dist: %lf\n", [classDefault distanceFrom:t]);
  // EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [s distanceFrom:s]);
  // EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [classDefault distanceFrom:anotherDefault]);
  // EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [null distanceFrom:zero]);
  // EXE-NEXT: Dist: 0.00
  printf("Dist: %lf\n", [s distanceFrom:cloned]);

  // Five shapes are allocated.
  // EXE: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NEXT: Dealloc
  // EXE-NOT: Dealloc
}
  return 0;
}
