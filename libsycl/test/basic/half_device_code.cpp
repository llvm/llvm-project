//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not=double

// Checks that sycl::half compiles to native half arithmetic in device code, and
// that compound assignment with double operands doesn't introduce double
// arithmetic. Also checks the operator result types and the constexpr
// numeric_limits in device compilation, in a kernel without a name.

#include "Inputs/half_common.hpp"

#include <sycl/sycl.hpp>

void test(sycl::queue Q, sycl::half *Ptr) {
  Q.single_task<class HalfKernel>([=] {
    sycl::half H = Ptr[0];
    H += 0.1;
    H *= 2;
    H = H / Ptr[1] - 1;
    Ptr[2] = -H;
    Ptr[3] = H < 65519 ? H : sycl::half(0.0f);
  });

  Q.single_task([=] {
    half_test::checkAllOperatorTypes();
    half_test::checkConstexprLimits();
    Ptr[4] = Ptr[0] * Ptr[1];
  });
}

// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}HalfKernel(
// CHECK-DAG: define {{.*}}spir_kernel void @{{.*}}UlvE

// CHECK-DAG: fadd half {{.*}}, 9.997550e-02
// CHECK-DAG: fmul half
// CHECK-DAG: fdiv half
// CHECK-DAG: fneg half
// CHECK-DAG: fcmp {{.*}} float
