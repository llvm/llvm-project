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

// CHECK:   [[OBJ:%.*]] = bitcast [[ONET:%.*]]* [[ONE:%.*]] to i8*
// CHECK:   [[SEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load ![[MD_NUM:[0-9]+]]
// CHECK:   [[CALL:%.*]] = call noundef nonnull align 1 %struct.Quad2* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to %struct.Quad2* (i8*, i8*)*)(i8* noundef [[OBJ]], i8* noundef [[SEL]])
// CHECK:   [[OBJ2:%.*]] = bitcast [[ONET]]* [[ZERO:%.*]] to i8*
// CHECK:   [[SEL2:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.2, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.Quad2*)*)(i8* noundef [[OBJ2]], i8* noundef [[SEL2]], %struct.Quad2* noundef nonnull align 1 [[CALL]])


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
// CHECK:   [[OBJ3:%.*]] = bitcast [[TWOT:%.*]]* [[ZERO1:%.*]] to i8*
// CHECK:   [[SEL3:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_.5, align 8, !invariant.load ![[MD_NUM]]
// CHECK:   call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, %struct.A*)*)(i8* noundef [[OBJ3]], i8* noundef [[SEL3]], %struct.A* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[ONE1]])
// CHECK:   store %struct.A* [[ONE1]], %struct.A** [[RESULT:%.*]], align 8
