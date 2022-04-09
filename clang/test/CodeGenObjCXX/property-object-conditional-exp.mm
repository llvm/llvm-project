// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

struct CGRect {
  char* origin;
  unsigned size;
};
typedef struct CGRect CGRect;

extern "C" bool CGRectIsEmpty(CGRect);

@interface Foo {
  CGRect out;
}
@property CGRect bounds;
- (CGRect) out;
@end


@implementation Foo

- (void)bar {
    CGRect dataRect;
    CGRect virtualBounds;

// CHECK: [[SRC:%.*]] = call { i8*, i32 } bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK-NEXT: bitcast
// CHECK-NEXT:getelementptr inbounds { i8*, i32 }, { i8*, i32 }* [[SRC:%.*]]
// CHECK-NEXT:extractvalue
// CHECK-NEXT:store
// CHECK-NEXT:getelementptr inbounds { i8*, i32 }, { i8*, i32 }* [[SRC:%.*]]
// CHECK-NEXT:extractvalue
// CHECK-NEXT:store
  dataRect = CGRectIsEmpty(virtualBounds) ? self.bounds : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? [self bounds] : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? virtualBounds : self.bounds;

  dataRect = CGRectIsEmpty(virtualBounds) ? self.out : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? [self out] : virtualBounds;
  dataRect = CGRectIsEmpty(virtualBounds) ? virtualBounds : self.out;
}

@dynamic bounds;
- (CGRect) out { return out; }
@end
