// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm %s -o -  | FileCheck %s

typedef double CGFloat;
struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct CGPoint CGPoint;



struct CGSize {
  CGFloat width;
  CGFloat height;
};
typedef struct CGSize CGSize;


struct CGRect {
  CGPoint origin;
  CGSize size;
};
typedef struct CGRect CGRect;

@interface UIView {
}
@property CGRect frame;
@end

@interface crashclass : UIView {

}

@end

@implementation crashclass
- (void)setFrame:(CGRect)frame
{
        super.frame = frame;
	[super setFrame:frame];
}

@end
// CHECK-NOT: declare void @objc_msgSendSuper2_stret
// CHECK: declare ptr @objc_msgSendSuper2
