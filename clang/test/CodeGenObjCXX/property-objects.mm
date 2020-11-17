// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -std=c++11 -emit-llvm -debug-info-kind=limited -o - | FileCheck %s

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
- (CGRect)extent;
- (void)dealloc;
@end

@implementation I
@synthesize position;
@synthesize bounds;
@synthesize frame;

// CHECK: define internal void @"\01-[I setPosition:]"
// CHECK: call nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %class.S* @_ZN1SaSERKS_
// CHECK-NEXT: ret void

// Don't attach debug locations to the prologue instructions. These were
// leaking over from the previous function emission by accident.
// CHECK: define internal void @"\01-[I setBounds:]"({{.*}} {
// CHECK-NOT: !dbg
// CHECK: call void @llvm.dbg.declare
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

// CHECK-LABEL: define i32 @main
// CHECK: call void @_ZN1SC1ERKS_(%class.S* {{[^,]*}} [[AGGTMP:%[a-zA-Z0-9\.]+]], %class.S* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) {{%[a-zA-Z0-9\.]+}})
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %class.S*)*)(i8* {{%[a-zA-Z0-9\.]+}}, i8* {{%[a-zA-Z0-9\.]+}}, %class.S* [[AGGTMP]])
// CHECK-NEXT: ret i32 0
int main() {
  I *i;
  S s1;
  i.position = s1;
  return 0;
}

// rdar://8379892
// CHECK-LABEL: define void @_Z1fP1A
// CHECK: call void @_ZN1XC1Ev(%struct.X* {{[^,]*}} [[LVTEMP:%[a-zA-Z0-9\.]+]])
// CHECK: call void @_ZN1XC1ERKS_(%struct.X* {{[^,]*}} [[AGGTMP:%[a-zA-Z0-9\.]+]], %struct.X* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[LVTEMP]])
// CHECK: call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.X*)*)({{.*}} %struct.X* [[AGGTMP]])
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

// rdar://21801088
//   Ensure that pseudo-objecet expressions that require the RHS to be
//   rewritten don't result in crashes or redundant emission of code.
struct B0 { long long x; };
struct B1 { long long x; }; B1 operator+(B1, B1);
struct B2 { B1 x; };
struct B3 { B3(); B1 x; operator B1(); };
@interface B
@property B0 b0;
@property B1 b1;
@property B2 b2;
@property B3 b3;
@end

int b_makeInt();

// Note that there's a promotion from int to long long, so
// the syntactic form of the RHS will be bogus.
void testB0(B *b) {
  b.b0 = { b_makeInt() };
}
void testB1(B *b) {
  b.b1 += { b_makeInt() };
}
// CHECK:    define void @_Z6testB0P1B([[B:%.*]]*
// CHECK:      [[BVAR:%.*]] = alloca [[B]]*, align 8
// CHECK:      [[TEMP:%.*]] = alloca [[B0:%.*]], align 8
// CHECK:      [[X:%.*]] = getelementptr inbounds [[B0]], [[B0]]* [[TEMP]], i32 0, i32 0
// CHECK-NEXT: [[T0:%.*]] = call i32 @_Z9b_makeIntv()
// CHECK-NEXT: [[T1:%.*]] = sext i32 [[T0]] to i64
// CHECK-NEXT: store i64 [[T1]], i64* [[X]], align 8
// CHECK:      load [[B]]*, [[B]]** [[BVAR]]
// CHECK-NOT:  call
// CHECK:      call void @llvm.memcpy
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      ret void

// CHECK:    define void @_Z6testB1P1B([[B]]*
// CHECK:      [[BVAR:%.*]] = alloca [[B]]*, align 8
// CHECK:      load [[B]]*, [[B]]** [[BVAR]]
// CHECK-NOT:  call
// CHECK:      [[T0:%.*]] = call i64 bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      store i64 [[T0]],
// CHECK-NOT:  call
// CHECK:      [[T0:%.*]] = call i32 @_Z9b_makeIntv()
// CHECK-NEXT: [[T1:%.*]] = sext i32 [[T0]] to i64
// CHECK-NEXT: store i64 [[T1]], i64* {{.*}}, align 8
// CHECK-NOT:  call
// CHECK:      [[T0:%.*]] = call i64 @_Zpl2B1S_
// CHECK-NOT:  call
// CHECK:      store i64 [[T0]],
// CHECK-NOT:  call
// CHECK:      call void @llvm.memcpy
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      ret void

// Another example of a conversion that needs to be applied
// in the semantic form.
void testB2(B *b) {
  b.b2 = { B3() };
}

// CHECK:    define void @_Z6testB2P1B([[B]]*
// CHECK:      [[BVAR:%.*]] = alloca [[B]]*, align 8
// CHECK:      call void @llvm.dbg.declare(
// CHECK:      call void @_ZN2B3C1Ev(
// CHECK-NEXT: [[T0:%.*]] = call i64 @_ZN2B3cv2B1Ev(
// CHECK-NOT:  call
// CHECK:      store i64 [[T0]],
// CHECK:      load [[B]]*, [[B]]** [[BVAR]]
// CHECK-NOT:  call
// CHECK:      call void @llvm.memcpy
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      ret void

// A similar test to B, but using overloaded function references.
struct C1 {
  int x;
  friend C1 operator+(C1, void(&)());
};
@interface C
@property void (*c0)();
@property C1 c1;
@end

void c_helper();
void c_helper(int);

void testC0(C *c) {
  c.c0 = c_helper;
  c.c0 = &c_helper;
}
// CHECK:    define void @_Z6testC0P1C([[C:%.*]]*
// CHECK:      [[CVAR:%.*]] = alloca [[C]]*, align 8
// CHECK:      load [[C]]*, [[C]]** [[CVAR]]
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend {{.*}} @_Z8c_helperv
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend {{.*}} @_Z8c_helperv
// CHECK-NOT:  call
// CHECK:      ret void

void testC1(C *c) {
  c.c1 += c_helper;
}
// CHECK:    define void @_Z6testC1P1C([[C]]*
// CHECK:      [[CVAR:%.*]] = alloca [[C]]*, align 8
// CHECK:      load [[C]]*, [[C]]** [[CVAR]]
// CHECK-NOT:  call
// CHECK:      [[T0:%.*]] = call i32 bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      store i32 [[T0]],
// CHECK-NOT:  call
// CHECK:      [[T0:%.*]] = call i32 @_Zpl2C1RFvvE({{.*}} @_Z8c_helperv
// CHECK-NOT:  call
// CHECK:      store i32 [[T0]],
// CHECK-NOT:  call
// CHECK:      call void @llvm.memcpy
// CHECK-NOT:  call
// CHECK:      call void bitcast {{.*}} @objc_msgSend
// CHECK-NOT:  call
// CHECK:      ret void
