// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks -fobjc-arc -fobjc-runtime-has-weak -DWEAK_SUPPORTED | FileCheck -check-prefix=ARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks | FileCheck -check-prefix=MRC %s

typedef int (^fp)();
fp f() { auto x = []{ return 3; }; return x; }

// ARC: %[[LAMBDACLASS:.*]] = type { i32 }

// MRC: @OBJC_METH_VAR_NAME{{.*}} = private unnamed_addr constant [5 x i8] c"copy\00"
// MRC: @OBJC_METH_VAR_NAME{{.*}} = private unnamed_addr constant [12 x i8] c"autorelease\00"
// MRC-LABEL: define{{.*}} ptr @_Z1fv(
// MRC-LABEL: define internal noundef ptr @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// MRC: store ptr @_NSConcreteStackBlock
// MRC: store ptr @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke"
// MRC: call noundef ptr @objc_msgSend
// MRC: call noundef ptr @objc_msgSend
// MRC: ret ptr

// ARC-LABEL: define{{.*}} ptr @_Z1fv(
// ARC-LABEL: define internal noundef ptr @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// ARC: store ptr @_NSConcreteStackBlock
// ARC: store ptr @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke"
// ARC: call ptr @llvm.objc.retainBlock
// ARC: call ptr @llvm.objc.autoreleaseReturnValue

typedef int (^fp)();
fp global;
void f2() { global = []{ return 3; }; }

// MRC: define{{.*}} void @_Z2f2v() [[NUW:#[0-9]+]] {
// MRC: store ptr @___Z2f2v_block_invoke,
// MRC-NOT: call
// MRC: ret void
// ("global" contains a dangling pointer after this function runs.)

// ARC: define{{.*}} void @_Z2f2v() [[NUW:#[0-9]+]] {
// ARC: store ptr @___Z2f2v_block_invoke,
// ARC: call ptr @llvm.objc.retainBlock
// ARC: call void @llvm.objc.release
// ARC-LABEL: define internal noundef i32 @___Z2f2v_block_invoke
// ARC: call noundef i32 @"_ZZ2f2vENK3$_0clEv

template <class T> void take_lambda(T &&lambda) { lambda(); }
void take_block(void (^block)()) { block(); }

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

// ARC: define{{.*}} void @_ZN13LambdaCapture4foo1ERi(ptr noundef nonnull align 4 dereferenceable(4) %{{.*}})
// ARC:   %[[CAPTURE0:.*]] = getelementptr inbounds nuw %[[LAMBDACLASS]], ptr %{{.*}}, i32 0, i32 0
// ARC:   store i32 %{{.*}}, ptr %[[CAPTURE0]]

// ARC: define internal void @"_ZZN13LambdaCapture4foo1ERiENK3$_0clEv"(ptr {{[^,]*}} %{{.*}})
// ARC:   %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, i32 }>
// ARC:   %[[CAPTURE1:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, i32 }>, ptr %[[BLOCK]], i32 0, i32 5
// ARC:   store i32 %{{.*}}, ptr %[[CAPTURE1]]

// ARC-LABEL: define internal void @"_ZZ10-[Foo foo]ENK3$_4clEv"(
// ARC-NOT: @llvm.objc.storeStrong(
// ARC: ret void

// ARC: define internal void @"___ZZN13LambdaCapture4foo1ERiENK3$_0clEv_block_invoke"
// ARC:   %[[CAPTURE2:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, i32 }>, ptr %{{.*}}, i32 0, i32 5
// ARC:   store i32 %{{.*}}, ptr %[[CAPTURE2]]

// ARC: define internal void @"___ZZN13LambdaCapture4foo1ERiENK3$_0clEv_block_invoke_2"(ptr noundef %{{.*}})
// ARC:   %[[CAPTURE3:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, i32 }>, ptr %{{.*}}, i32 0, i32 5
// ARC:   %[[V1:.*]] = load i32, ptr %[[CAPTURE3]]
// ARC:   store i32 %[[V1]], ptr @_ZN13LambdaCapture1iE

namespace LambdaCapture {
  int i;
  void foo1(int &a) {
    auto lambda = [a]{
      auto block1 = ^{
        auto block2 = ^{
          i = a;
        };
        block2();
      };
      block1();
    };
    lambda();
  }
}

// ARC-LABEL: define linkonce_odr noundef ptr @_ZZNK13StaticMembersIfE1fMUlvE_clEvENKUlvE_cvU13block_pointerFivEEv

// Check lines for BlockInLambda test below
// ARC-LABEL: define internal noundef i32 @___ZZN13BlockInLambda1X1fEvENKUlvE_clEv_block_invoke
// ARC: [[Y:%.*]] = getelementptr inbounds nuw %"struct.BlockInLambda::X", ptr {{.*}}, i32 0, i32 1
// ARC-NEXT: [[YVAL:%.*]] = load i32, ptr [[Y]], align 4
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

@interface NSObject @end
@interface Foo : NSObject @end
@implementation Foo
- (void)foo {
  [&] {
    ^{ (void)self; }();
  }();
}
@end

// Check that the delegating invoke function doesn't destruct the Weak object
// that is passed.

// ARC-LABEL: define internal void @"_ZZN14LambdaDelegate4testEvEN3$_08__invokeENS_4WeakE"(
// ARC: call void @"_ZZN14LambdaDelegate4testEvENK3$_0clENS_4WeakE"(
// ARC-NEXT: ret void

// ARC-LABEL: define internal void @"_ZZN14LambdaDelegate4testEvENK3$_0clENS_4WeakE"(
// ARC: call void @_ZN14LambdaDelegate4WeakD1Ev(

#ifdef WEAK_SUPPORTED

namespace LambdaDelegate {

struct Weak {
  __weak id x;
};

void test() {
  void (*p)(Weak) = [](Weak a) { };
}

};

#endif

// ARC: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
// MRC: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
