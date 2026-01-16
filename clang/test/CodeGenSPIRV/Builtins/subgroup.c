// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device -x c++ %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -O1 -triple spirv64 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -O1 -triple spirv32 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

#if defined(__cplusplus)
typedef bool _Bool;
#endif
typedef unsigned __attribute__((ext_vector_type(4))) int4;

// CHECK: @{{.*}}test_subgroup_ballot{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call <4 x i32> @llvm.spv.wave.ballot(i1 %i)
[[clang::sycl_external]] int4 test_subgroup_ballot(_Bool i) {
    return __builtin_spirv_subgroup_ballot(i);
}

// CHECK: @{{.*}}test_subgroup_shuffle{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call float @llvm.spv.wave.readlane.f32(float %f, i32 %i)
//
[[clang::sycl_external]] float test_subgroup_shuffle(float f, int i) {
    return __builtin_spirv_subgroup_shuffle(f, i);
}
