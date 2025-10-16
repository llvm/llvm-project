// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -std=c++14 -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK: %[[S:.*]] = type { i32 }
// CHECK: %[[CLASS_ANON_2:.*]] = type { ptr }
// CHECK: %[[CLASS_ANON_3:.*]] = type { %[[S]] }

// CHECK: %[[BLOCK_CAPTURED0:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, ptr %[[BLOCK:.*]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds nuw %[[LAMBDA_CLASS:.*]], ptr %[[THIS:.*]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[V0]], align 8
// CHECK: store ptr %[[V1]], ptr %[[BLOCK_CAPTURED0]], align 8
// CHECK: %[[BLOCK_CAPTURED1:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr }>, ptr %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V2:.*]] = getelementptr inbounds nuw %[[LAMBDA_CLASS]], ptr %[[THIS]], i32 0, i32 1
// CHECK: %[[V3:.*]] = load ptr, ptr %[[V2]], align 8
// CHECK: store ptr %[[V3]], ptr %[[BLOCK_CAPTURED1]], align 8

void foo1(int &, int &);

void block_in_lambda(int &s1, int &s2) {
  auto lambda = [&s1, &s2]() {
    auto block = ^{
      foo1(s1, s2);
    };
    block();
  };

  lambda();
}

namespace CaptureByReference {

id getObj();
void use(id);

// Block copy/dispose helpers aren't needed because 'a' is captured by
// reference.

// CHECK-LABEL: define{{.*}} void @_ZN18CaptureByReference5test0Ev(
// CHECK-LABEL: define internal void @"_ZZN18CaptureByReference5test0EvENK3$_0clEv"(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @"__block_descriptor_40_e5_v8\01?0ls32l8", ptr %[[BLOCK_DESCRIPTOR]], align 8

void test0() {
  id a = getObj();
  [&]{ ^{ a = 0; }(); }();
}

// Block copy/dispose helpers shouldn't have to retain/release 'a' because it
// is captured by reference.

// CHECK-LABEL: define{{.*}} void @_ZN18CaptureByReference5test1Ev(
// CHECK-LABEL: define internal void @"_ZZN18CaptureByReference5test1EvENK3$_0clEv"(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 4
// CHECK: store ptr @"__block_descriptor_56_8_32s40s_e5_v8\01?0l", ptr %[[BLOCK_DESCRIPTOR]], align 8

void test1() {
  id a = getObj(), b = getObj(), c = getObj();
  [&a, b, c]{ ^{ a = 0; use(b); use(c); }(); }();
}

struct S {
  int val() const;
  int a;
  S();
  S(const S&);
  S &operator=(const S&);
  S(S&&);
  S &operator=(S&&);
};

S getS();

// CHECK: define internal noundef i32 @"_ZZN18CaptureByReference5test2EvENK3$_0clIiEEDaT_"(ptr {{[^,]*}} %{{.*}}, i32 noundef %{{.*}})
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, {{.*}}, ptr }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, {{.*}}, ptr }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds nuw %[[CLASS_ANON_2]], ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[V1:.*]] = load ptr, ptr %[[V0]], align 8
// CHECK: store ptr %[[V1]], ptr %[[BLOCK_CAPTURED]], align 8

int test2() {
  S s;
  auto fn = [&](const auto a){
    return ^{
      return s.val();
    }();
  };
  return fn(123);
}

// CHECK: define internal noundef i32 @"_ZZN18CaptureByReference5test3EvENK3$_0clIiEEDaT_"(ptr {{[^,]*}} %{{.*}}, i32 noundef %{{.*}})
// CHECK: %[[BLOCK:.*]] = alloca <{ ptr, i32, i32, ptr, ptr, %[[S]] }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, %[[S]] }>, ptr %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds nuw %[[CLASS_ANON_3]], ptr %{{.*}}, i32 0, i32 0
// CHECK: call void @_ZN18CaptureByReference1SC1ERKS0_(ptr {{[^,]*}} %[[BLOCK_CAPTURED]], ptr {{.*}} %[[V0]])

int test3() {
  const S &s = getS();
  auto fn = [=](const auto a){
    return ^{
      return s.val();
    }();
  };
  return fn(123);
}

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_8_32s40s(
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: %[[V4:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[V5:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[BLOCKCOPY_SRC:.*]] = load ptr, ptr %[[V4]], align 8
// CHECK: store ptr null, ptr %[[V5]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V5]], ptr %[[BLOCKCOPY_SRC]])
// CHECK: %[[V6:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 6
// CHECK: %[[V7:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 6
// CHECK: %[[BLOCKCOPY_SRC2:.*]] = load ptr, ptr %[[V6]], align 8
// CHECK: store ptr null, ptr %[[V7]], align 8
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V7]], ptr %[[BLOCKCOPY_SRC2]])
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_8_32s40s(
// CHECK: %[[V2:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 5
// CHECK: %[[V3:.*]] = getelementptr inbounds nuw <{ ptr, i32, i32, ptr, ptr, ptr, ptr, ptr }>, ptr %{{.*}}, i32 0, i32 6
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V3]], ptr null)
// CHECK: call void @llvm.objc.storeStrong(ptr %[[V2]], ptr null)
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: ret void

}
