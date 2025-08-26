// RUN: %clang_cc1 %s -triple arm64-apple-macosx -emit-llvm -fcxx-exceptions -fexceptions -std=c++23    -fsized-deallocation    -faligned-allocation -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -fno-sized-deallocation    -faligned-allocation -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx -emit-llvm -fcxx-exceptions -fexceptions -std=c++23    -fsized-deallocation -fno-aligned-allocation -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple arm64-apple-macosx -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -fno-aligned-allocation -fno-sized-deallocation -o - | FileCheck %s

namespace std {
  template <class T> struct type_identity {};
  enum class align_val_t : __SIZE_TYPE__ {};
}

using size_t = __SIZE_TYPE__;
struct Context;
struct S1 {
  S1();
  int i;
};

void *operator new(std::type_identity<S1>, size_t, std::align_val_t, Context&);
void operator delete(std::type_identity<S1>, void*, size_t, std::align_val_t, Context&); // #1
void operator delete(std::type_identity<S1>, void*, size_t, std::align_val_t);

struct S2 {
  S2();
  int i;
  template<typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t, Context&);
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t, Context&); // #2
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #3
};

struct S3 {
  S3();
  int i;
  template<typename T> void *operator new(std::type_identity<T>, size_t, std::align_val_t);
  template<typename T> void operator delete(std::type_identity<T>, void*, size_t, std::align_val_t); // #4
};

extern "C" void test_s1(Context& Ctx) {
  S1 *s1 = new (Ctx) S1;
  delete s1;
}

// CHECK-LABEL: test_s1
// CHECK: [[S1_NEW:%.*]] = call noundef ptr @_ZnwSt13type_identityI2S1EmSt11align_val_tR7Context({{.*}},{{.*}} 4,{{.*}} 4, ptr noundef nonnull align 1 [[S1_NEW_CONTEXT:%.*]])
// CHECK: call void @_ZdlSt13type_identityI2S1EPvmSt11align_val_t({{.*}}, ptr noundef %2,{{.*}} 4,{{.*}} 4)
// CHECK: call void @_ZdlSt13type_identityI2S1EPvmSt11align_val_tR7Context({{.*}}, ptr noundef [[S1_NEW]],{{.*}} 4,{{.*}} 4, ptr noundef nonnull align 1 [[S1_NEW_CONTEXT]])

extern "C" void test_s2(Context& Ctx) {
  S2 *s2_1 = new (Ctx) S2;
  delete s2_1;
  S2 *s2_2 = new (std::align_val_t(128), Ctx) S2;
  delete s2_2;
}

// CHECK-LABEL: test_s2
// CHECK: [[S2_NEW1:%.*]] = call noundef ptr @_ZN2S2nwIS_EEPvSt13type_identityIT_EmSt11align_val_tR7Context({{.*}},{{.*}} 4,{{.*}} 4,{{.*}} [[S2_NEW1_CONTEXT:%.*]])
// CHECK: call void @_ZN2S2dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}},{{.*}} 4,{{.*}} 4)
// CHECK: [[S2_NEW2:%.*]] = call noundef ptr @_ZN2S2nwIS_EEPvSt13type_identityIT_EmSt11align_val_tR7Context({{.*}},{{.*}} 4,{{.*}} 128,{{.*}} [[S2_NEW2_CONTEXT:%.*]])
// CHECK: call void @_ZN2S2dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}},{{.*}} 4,{{.*}} 4)
// CHECK: call void @_ZN2S2dlIS_EEvSt13type_identityIT_EPvmSt11align_val_tR7Context({{.*}}, {{.*}} [[S2_NEW1]],{{.*}} 4,{{.*}} 4,{{.*}} [[S2_NEW1_CONTEXT]])
// CHECK: call void @_ZN2S2dlIS_EEvSt13type_identityIT_EPvmSt11align_val_tR7Context({{.*}}, {{.*}} [[S2_NEW2]],{{.*}} 4,{{.*}} 128,{{.*}} [[S2_NEW2_CONTEXT]])

extern "C" void test_s3(Context& Ctx) {
  S3 *s3_1 = new S3;
  delete s3_1;
  S3 *s3_2 = new (std::align_val_t(128)) S3;
  delete s3_2;
}

// CHECK-LABEL: test_s3
// CHECK: [[S3_NEW1:%.*]] = call noundef ptr @_ZN2S3nwIS_EEPvSt13type_identityIT_EmSt11align_val_t({{.*}},{{.*}} 4,{{.*}} 4)
// CHECK: call void @_ZN2S3dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}},{{.*}} 4,{{.*}} 4) 
// CHECK: [[S3_NEW2:%.*]] = call noundef ptr @_ZN2S3nwIS_EEPvSt13type_identityIT_EmSt11align_val_t({{.*}},{{.*}} 4,{{.*}} 128)
// CHECK: call void @_ZN2S3dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}},{{.*}} 4,{{.*}} 4)
// CHECK: call void @_ZN2S3dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}}[[S3_NEW1]],{{.*}} 4,{{.*}} 4)
// CHECK: call void @_ZN2S3dlIS_EEvSt13type_identityIT_EPvmSt11align_val_t({{.*}},{{.*}}[[S3_NEW2]],{{.*}} 4,{{.*}} 128)
