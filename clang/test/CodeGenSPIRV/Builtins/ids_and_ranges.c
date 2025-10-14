// RUN: %clang_cc1 -O1 -triple spirv64 -fsycl-is-device -x c++ %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -O1 -triple spirv64 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK64
// RUN: %clang_cc1 -O1 -triple spirv32 -cl-std=CL3.0 -x cl %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK32

// CHECK: @{{.*}}test_num_workgroups{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.num.workgroups.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.num.workgroups.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_num_workgroups() {
    return __builtin_spirv_num_workgroups(0);
}

// CHECK: @{{.*}}test_workgroup_size{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.workgroup.size.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.workgroup.size.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_workgroup_size() {
    return __builtin_spirv_workgroup_size(0);
}

// CHECK: @{{.*}}test_workgroup_id{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.group.id.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.group.id.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_workgroup_id() {
    return __builtin_spirv_workgroup_id(0);
}

// CHECK: @{{.*}}test_local_invocation_id{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.thread.id.in.group.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_local_invocation_id() {
    return __builtin_spirv_local_invocation_id(0);
}

// CHECK: @{{.*}}test_global_invocation_id{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.thread.id.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.thread.id.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_global_invocation_id() {
    return __builtin_spirv_global_invocation_id(0);
}

// CHECK: @{{.*}}test_global_size{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.global.size.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.global.size.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_global_size() {
    return __builtin_spirv_global_size(0);
}

// CHECK: @{{.*}}test_global_offset{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK64-NEXT:    tail call i64 @llvm.spv.global.offset.i64(i32 0)
// CHECK32-NEXT:    tail call i32 @llvm.spv.global.offset.i32(i32 0)
//
[[clang::sycl_external]] unsigned int test_global_offset() {
    return __builtin_spirv_global_offset(0);
}

// CHECK: @{{.*}}test_subgroup_size{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call i32 @llvm.spv.subgroup.size()
//
[[clang::sycl_external]] unsigned int test_subgroup_size() {
    return __builtin_spirv_subgroup_size();
}

// CHECK: @{{.*}}test_subgroup_max_size{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call i32 @llvm.spv.subgroup.max.size()
//
[[clang::sycl_external]] unsigned int test_subgroup_max_size() {
    return __builtin_spirv_subgroup_max_size();
}

// CHECK: @{{.*}}test_num_subgroups{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call i32 @llvm.spv.num.subgroups()
//
[[clang::sycl_external]] unsigned int test_num_subgroups() {
    return __builtin_spirv_num_subgroups();
}

// CHECK: @{{.*}}test_subgroup_id{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call i32 @llvm.spv.subgroup.id()
//
[[clang::sycl_external]] unsigned int test_subgroup_id() {
    return __builtin_spirv_subgroup_id();
}

// CHECK: @{{.*}}test_subgroup_local_invocation_id{{.*}}(
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    tail call i32 @llvm.spv.subgroup.local.invocation.id()
//
[[clang::sycl_external]] unsigned int test_subgroup_local_invocation_id() {
    return __builtin_spirv_subgroup_local_invocation_id();
}
