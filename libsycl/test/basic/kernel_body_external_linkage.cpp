//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A kernel body in a function with external linkage is emitted for the host
// too, so every work-item query it makes has to link without a device. The
// link step below is the check. The FileCheck step then confirms that the
// device still reads the real builtins, which a host fallback applied
// unconditionally would silently replace with constants.

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

#include <cstddef>

// External linkage on purpose: a static or inline definition reachable only
// from the kernel is not emitted for the host, and would not test anything.
void writeWorkItemIds(sycl::nd_item<1> It, std::size_t *Out) {
  // CHECK-DAG: call noundef i64 @llvm.spv.thread.id.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.thread.id.in.group.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.group.id.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.global.size.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.workgroup.size.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.num.workgroups.i64(i32 0)
  // CHECK-DAG: call noundef i64 @llvm.spv.global.offset.i64(i32 0)
  Out[It.get_global_linear_id()] =
      It.get_global_id(0) + It.get_local_id(0) + It.get_group(0) +
      It.get_global_range(0) + It.get_local_range(0) + It.get_group_range(0);

  sycl::group<1> G = It.get_group();
  Out[G.get_local_linear_id()] = G.get_group_id(0) + G.get_local_id(0) +
                                 G.get_local_range(0) + G.get_group_range(0);

  // CHECK-DAG: call noundef i32 @llvm.spv.subgroup.id()
  // CHECK-DAG: call noundef i32 @llvm.spv.subgroup.local.invocation.id()
  // CHECK-DAG: call noundef i32 @llvm.spv.subgroup.size()
  // CHECK-DAG: call noundef i32 @llvm.spv.num.subgroups()
  // CHECK-DAG: call noundef i32 @llvm.spv.subgroup.max.size()
  sycl::sub_group SG = It.get_sub_group();
  Out[SG.get_local_linear_id()] =
      SG.get_group_id()[0] + SG.get_local_id()[0] + SG.get_local_range()[0] +
      SG.get_group_range()[0] + SG.get_max_local_range()[0];

  // The scope and semantics operands are covered by
  // group_barrier_device_code.cpp; here the call only has to survive the
  // device-only guard.
  // CHECK-DAG: call spir_func void @_Z22__spirv_ControlBarrierjjj(
  sycl::group_barrier(G);
  sycl::group_barrier(SG);
}

int main() {
  sycl::queue Q;
  std::size_t *Data = sycl::malloc_device<std::size_t>(8, Q);
  Q.parallel_for<class ExternalKernelBody>(
      sycl::nd_range<1>{8, 4},
      [=](sycl::nd_item<1> It) { writeWorkItemIds(It, Data); });
  Q.wait();
  sycl::free(Data, Q);
  return 0;
}
