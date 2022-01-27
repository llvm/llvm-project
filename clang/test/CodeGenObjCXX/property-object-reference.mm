// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - | FileCheck %s
// rdar://10188258

struct Foo {int i;};
static Foo gFoo;


@interface ObjCTest  { }
@property (nonatomic, readonly) Foo& FooRefProperty;
@property (nonatomic) Foo  FooProperty;
- (Foo &) FooProperty;
- (void)setFooProperty : (Foo &) arg;
@end


@implementation ObjCTest
@dynamic FooRefProperty;

-(void) test {
    Foo& f = self.FooRefProperty;
    Foo& f1 = self.FooProperty;
}
- (Foo &) FooProperty { return gFoo; }
- (void)setFooProperty : (Foo &) arg {  };
@end

// CHECK: [[T0:%.*]] = load {{%.*}} [[S0:%.*]]
// CHECK:  [[T2:%.*]]  = bitcast {{%.*}} [[T0]] to i8*
// CHECK: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK:  @objc_msgSend
// CHECK: [[R0:%.*]] = load {{%.*}} [[U0:%.*]]
// CHECK:  [[R2:%.*]]  = bitcast {{%.*}} [[R0]] to i8*
// CHECK: load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK:  @objc_msgSend

