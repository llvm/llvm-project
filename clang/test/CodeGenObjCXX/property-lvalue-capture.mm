// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://15118128

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

// CHECK:   [[TWO:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load ![[MD_NUM:[0-9]+]]
// CHECK:   [[THREE:%.*]] = bitcast [[ONET:%.*]]* [[ONE:%.*]] to i8*
// CHECK:   [[CALL:%.*]] = call nonnull align 1 %struct.Quad2* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %struct.Quad2* (i8*, i8*)*)(i8* [[THREE]], i8* [[TWO]])
// CHECK:   [[FOUR:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.2, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   [[FIVE:%.*]] = bitcast [[ONET]]* [[ZERO:%.*]] to i8*
// CHECK:   call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.Quad2*)*)(i8* [[FIVE]], i8* [[FOUR]], %struct.Quad2* nonnull align 1 [[CALL]])


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

// CHECK:   [[ONE1:%.*]] = load %struct.A*, %struct.A** [[AADDR:%.*]], align 8
// CHECK:   [[TWO1:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.5, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   [[THREE1:%.*]] = bitcast [[TWOT:%.*]]* [[ZERO1:%.*]] to i8*
// CHECK:   call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.A*)*)(i8* [[THREE1]], i8* [[TWO1]], %struct.A* nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[ONE1]])
// CHECK:   store %struct.A* [[ONE1]], %struct.A** [[RESULT:%.*]], align 8
