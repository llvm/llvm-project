// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// CHECK-NOT: callq	_objc_msgSend_stret
// CHECK: call void @_ZN1SC1ERKS_
// CHECK: call %class.S* @_ZN1SaSERKS_
// CHECK: call %struct.CGRect* @_ZN6CGRectaSERKS_

class S {
public:
	S& operator = (const S&);
	S (const S&);
	S ();
};

struct CGRect {
	CGRect & operator = (const CGRect &);
};

@interface I {
  S position;
  CGRect bounds;
}
@property(assign, nonatomic) S position;
@property CGRect bounds;
@property CGRect frame;
- (void)setFrame:(CGRect)frameRect;
- (CGRect)frame;
- (void) initWithOwner;
- (struct CGRect)extent;
- (void)dealloc;
@end

@implementation I
@synthesize position;
@synthesize bounds;
@synthesize frame;
- (void)setFrame:(CGRect)frameRect {}
- (CGRect)frame {return bounds;}

- (void)initWithOwner {
  I* _labelLayer;
  CGRect labelLayerFrame = self.bounds;
  labelLayerFrame = self.bounds;
  _labelLayer.frame = labelLayerFrame;
}
// rdar://8366604
- (void)dealloc
  {
      CGRect cgrect = self.extent;
  }
- (struct CGRect)extent {return bounds;}
@end

int main() {
  I *i;
  S s1;
  i.position = s1;
  return 0;
}

// rdar://8379892
// CHECK: define void @_Z1fP1A
// CHECK: @objc_msgSend to void
struct X {
  X();
  X(const X&);
  ~X();
};

@interface A {
  X xval;
}
- (X)x;
- (void)setX:(X)x;
@end

void f(A* a) {
  a.x = X();
}
