// RUN: %clang_cc1 -triple=x86_64-apple-darwin10 -emit-llvm -std=c++14 -fblocks -fobjc-arc -o - %s | FileCheck %s

// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }
// CHECK: %[[S:.*]] = type { i32 }
// CHECK: %[[CLASS_ANON_2:.*]] = type { %[[S]]* }
// CHECK: %[[CLASS_ANON_3:.*]] = type { %[[S]] }

// CHECK: %[[BLOCK_CAPTURED0:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>* %[[BLOCK:.*]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds %[[LAMBDA_CLASS:.*]], %[[LAMBDA_CLASS]]* %[[THIS:.*]], i32 0, i32 0
// CHECK: %[[V1:.*]] = load i32*, i32** %[[V0]], align 8
// CHECK: store i32* %[[V1]], i32** %[[BLOCK_CAPTURED0]], align 8
// CHECK: %[[BLOCK_CAPTURED1:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>, <{ i8*, i32, i32, i8*, %struct.__block_descriptor*, i32*, i32* }>* %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V2:.*]] = getelementptr inbounds %[[LAMBDA_CLASS]], %[[LAMBDA_CLASS]]* %[[THIS]], i32 0, i32 1
// CHECK: %[[V3:.*]] = load i32*, i32** %[[V2]], align 8
// CHECK: store i32* %[[V3]], i32** %[[BLOCK_CAPTURED1]], align 8

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

// CHECK-LABEL: define void @_ZN18CaptureByReference5test0Ev(
// CHECK-LABEL: define internal void @"_ZZN18CaptureByReference5test0EvENK3$_3clEv"(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8** }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i64 }* @"__block_descriptor_40_e5_v8\01?0ls32l8" to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8

void test0() {
  id a = getObj();
  [&]{ ^{ a = 0; }(); }();
}

// Block copy/dispose helpers shouldn't have to retain/release 'a' because it
// is captured by reference.

// CHECK-LABEL: define void @_ZN18CaptureByReference5test1Ev(
// CHECK-LABEL: define internal void @"_ZZN18CaptureByReference5test1EvENK3$_4clEv"(
// CHECK: %[[BLOCK_DESCRIPTOR:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 4
// CHECK: store %[[STRUCT_BLOCK_DESCRIPTOR]]* bitcast ({ i64, i64, i8*, i8*, i8*, i64 }* @"__block_descriptor_56_8_32s40s_e5_v8\01?0l" to %[[STRUCT_BLOCK_DESCRIPTOR]]*), %[[STRUCT_BLOCK_DESCRIPTOR]]** %[[BLOCK_DESCRIPTOR]], align 8

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

// CHECK: define internal i32 @"_ZZN18CaptureByReference5test2EvENK3$_1clIiEEDaT_"(%[[CLASS_ANON_2]]* {{[^,]*}} %{{.*}}, i32 %{{.*}})
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %{{.*}}, %[[S]]* }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %{{.*}}, %[[S]]* }>, <{ i8*, i32, i32, i8*, %{{.*}}, %[[S]]* }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds %[[CLASS_ANON_2]], %[[CLASS_ANON_2]]* %{{.*}}, i32 0, i32 0
// CHECK: %[[V1:.*]] = load %[[S]]*, %[[S]]** %[[V0]], align 8
// CHECK: store %[[S]]* %[[V1]], %[[S]]** %[[BLOCK_CAPTURED]], align 8

int test2() {
  S s;
  auto fn = [&](const auto a){
    return ^{
      return s.val();
    }();
  };
  return fn(123);
}

// CHECK: define internal i32 @"_ZZN18CaptureByReference5test3EvENK3$_2clIiEEDaT_"(%[[CLASS_ANON_3]]* {{[^,]*}} %{{.*}}, i32 %{{.*}})
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %{{.*}}*, %[[S]] }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %{{.*}}*, %[[S]] }>, <{ i8*, i32, i32, i8*, %{{.*}}*, %[[S]] }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = getelementptr inbounds %[[CLASS_ANON_3]], %[[CLASS_ANON_3]]* %{{.*}}, i32 0, i32 0
// CHECK: call void @_ZN18CaptureByReference1SC1ERKS0_(%[[S]]* {{[^,]*}} %[[BLOCK_CAPTURED]], %[[S]]* {{.*}} %[[V0]])

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
// CHECK: %[[V4:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 5
// CHECK: %[[V5:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 5
// CHECK: %[[BLOCKCOPY_SRC:.*]] = load i8*, i8** %[[V4]], align 8
// CHECK: store i8* null, i8** %[[V5]], align 8
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V5]], i8* %[[BLOCKCOPY_SRC]])
// CHECK: %[[V6:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 6
// CHECK: %[[V7:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 6
// CHECK: %[[BLOCKCOPY_SRC2:.*]] = load i8*, i8** %[[V6]], align 8
// CHECK: store i8* null, i8** %[[V7]], align 8
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V7]], i8* %[[BLOCKCOPY_SRC2]])
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_8_32s40s(
// CHECK: %[[V2:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 5
// CHECK: %[[V3:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8*, i8** }>* %{{.*}}, i32 0, i32 6
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V3]], i8* null)
// CHECK: call void @llvm.objc.storeStrong(i8** %[[V2]], i8* null)
// CHECK-NOT: call void @llvm.objc.storeStrong(
// CHECK: ret void

}
