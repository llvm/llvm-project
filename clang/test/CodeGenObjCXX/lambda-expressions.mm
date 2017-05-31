// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks -fobjc-arc | FileCheck -check-prefix=ARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks | FileCheck -check-prefix=MRC %s

typedef int (^fp)();
fp f() { auto x = []{ return 3; }; return x; }

// MRC: @OBJC_METH_VAR_NAME{{.*}} = private global [5 x i8] c"copy\00"
// MRC: @OBJC_METH_VAR_NAME{{.*}} = private global [12 x i8] c"autorelease\00"
// MRC-LABEL: define i32 ()* @_Z1fv(
// MRC-LABEL: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// MRC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// MRC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke" to i8*)
// MRC: call i32 ()* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: call i32 ()* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: ret i32 ()*

// ARC-LABEL: define i32 ()* @_Z1fv(
// ARC-LABEL: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// ARC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// ARC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke" to i8*)
// ARC: call i8* @objc_retainBlock
// ARC: call i8* @objc_autoreleaseReturnValue

typedef int (^fp)();
fp global;
void f2() { global = []{ return 3; }; }

// MRC: define void @_Z2f2v() [[NUW:#[0-9]+]] {
// MRC: store i8* bitcast (i32 (i8*)* @___Z2f2v_block_invoke to i8*),
// MRC-NOT: call
// MRC: ret void
// ("global" contains a dangling pointer after this function runs.)

// ARC: define void @_Z2f2v() [[NUW:#[0-9]+]] {
// ARC: store i8* bitcast (i32 (i8*)* @___Z2f2v_block_invoke to i8*),
// ARC: call i8* @objc_retainBlock
// ARC: call void @objc_release
// ARC-LABEL: define internal i32 @___Z2f2v_block_invoke
// ARC: call i32 @"_ZZ2f2vENK3$_1clEv

template <class T> void take_lambda(T &&lambda) { lambda(); }
void take_block(void (^block)()) { block(); }

// rdar://13800041
@interface A
- (void) test;
@end
@interface B : A @end
@implementation B
- (void) test {
  take_block(^{
      take_lambda([=]{
          take_block(^{
              take_lambda([=] {
                  [super test];
              });
          });
      });
   });
}
@end

// ARC-LABEL: define linkonce_odr i32 ()* @_ZZNK13StaticMembersIfE1fMUlvE_clEvENKUlvE_cvU13block_pointerFivEEv

// Check lines for BlockInLambda test below
// ARC-LABEL: define internal i32 @___ZZN13BlockInLambda1X1fEvENKUlvE_clEv_block_invoke
// ARC: [[Y:%.*]] = getelementptr inbounds %"struct.BlockInLambda::X", %"struct.BlockInLambda::X"* {{.*}}, i32 0, i32 1
// ARC-NEXT: [[YVAL:%.*]] = load i32, i32* [[Y]], align 4
// ARC-NEXT: ret i32 [[YVAL]]

typedef int (^fptr)();
template<typename T> struct StaticMembers {
  static fptr f;
};
template<typename T>
fptr StaticMembers<T>::f = [] { auto f = []{return 5;}; return fptr(f); }();
template fptr StaticMembers<float>::f;

namespace BlockInLambda {
  struct X {
    int x,y;
    void f() {
      [this]{return ^{return y;}();}();
    };
  };
  void g(X& x) {
    x.f();
  };
}


// ARC: attributes [[NUW]] = { nounwind{{.*}} }
// MRC: attributes [[NUW]] = { nounwind{{.*}} }
