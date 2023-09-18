// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

template <typename T> struct Quad2 {
  Quad2() {}
};

typedef Quad2<double> Quad2d;

@interface Root @end

@interface PAGeometryFrame
- (const Quad2d &)quad;
- (void)setQuad:(const Quad2d &)quad;
@end

@interface PA2DScaleTransform  : Root
@end

@implementation PA2DScaleTransform
- (void)transformFrame:(PAGeometryFrame *)frame {
 PAGeometryFrame *result;
 result.quad  = frame.quad;
}
@end

// CHECK:   [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load ![[MD_NUM:[0-9]+]]
// CHECK:   [[CALL:%.*]] = call noundef nonnull align 1 ptr @objc_msgSend(ptr noundef [[ONE:%.*]], ptr noundef [[SEL]])
// CHECK:   [[SEL2:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.2, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   call void @objc_msgSend(ptr noundef [[ZERO:%.*]], ptr noundef [[SEL2]], ptr noundef nonnull align 1 [[CALL]])


struct A {
 void *ptr;
 A();
 A(const A &);
 ~A();
};

@interface C
- (void) setProp: (const A&) value;
@end
void test(C *c, const A &a) {
 const A &result = c.prop = a;
}

// CHECK:   [[ONE1:%.*]] = load ptr, ptr [[AADDR:%.*]], align 8
// CHECK:   [[SEL3:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.5, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   call void @objc_msgSend(ptr noundef [[ZERO1:%.*]], ptr noundef [[SEL3]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[ONE1]])
// CHECK:   store ptr [[ONE1]], ptr [[RESULT:%.*]], align 8
