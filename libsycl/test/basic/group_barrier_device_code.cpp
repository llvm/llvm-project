// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Checks the execution scope, memory scope and memory semantics operands that
// group_barrier passes to __spirv_ControlBarrier.

#include <sycl/sycl.hpp>

void test(sycl::queue Q) {
  Q.parallel_for<class barrier_kernel>(
      sycl::nd_range<1>{8, 4}, [=](sycl::nd_item<1> It) {
        // Execution scope is Workgroup (2), memory scope defaults to the
        // group's fence_scope, i.e. Workgroup (2).
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 2, i32 noundef 400)
        sycl::group_barrier(It.get_group());
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 4, i32 noundef 400)
        sycl::group_barrier(It.get_group(), sycl::memory_scope::work_item);
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 3, i32 noundef 400)
        sycl::group_barrier(It.get_group(), sycl::memory_scope::sub_group);
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 2, i32 noundef 400)
        sycl::group_barrier(It.get_group(), sycl::memory_scope::work_group);
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 1, i32 noundef 400)
        sycl::group_barrier(It.get_group(), sycl::memory_scope::device);
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 2, i32 noundef 0, i32 noundef 400)
        sycl::group_barrier(It.get_group(), sycl::memory_scope::system);

        // Execution scope is Subgroup (3), memory scope defaults to the
        // sub-group's fence_scope, i.e. Subgroup (3).
        // CHECK: call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef
        // 3, i32 noundef 3, i32 noundef 400)
        sycl::group_barrier(It.get_sub_group());
      });
}
