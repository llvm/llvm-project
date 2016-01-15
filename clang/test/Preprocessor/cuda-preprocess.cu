// Tests CUDA compilation with -E.

// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

#ifndef __CUDA_ARCH__
#define PREPROCESSED_AWAY
clang_unittest_no_arch PREPROCESSED_AWAY
#else
clang_unittest_cuda_arch __CUDA_ARCH__
#endif

// CHECK-NOT: PREPROCESSED_AWAY

// RUN: %clang -E -target x86_64-linux-gnu --cuda-gpu-arch=sm_20 %s 2>&1 \
// RUN:   | FileCheck -check-prefix NOARCH %s
// RUN: %clang -E -target x86_64-linux-gnu --cuda-gpu-arch=sm_20 --cuda-host-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix NOARCH %s
// NOARCH: clang_unittest_no_arch

// RUN: %clang -E -target x86_64-linux-gnu --cuda-gpu-arch=sm_20 --cuda-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix SM20 %s
// SM20: clang_unittest_cuda_arch 200

// RUN: %clang -E -target x86_64-linux-gnu --cuda-gpu-arch=sm_30 --cuda-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix SM30 %s
// SM30: clang_unittest_cuda_arch 300

// RUN: %clang -E -target x86_64-linux-gnu --cuda-gpu-arch=sm_20 --cuda-gpu-arch=sm_30 \
// RUN:   --cuda-device-only %s 2>&1 \
// RUN:   | FileCheck -check-prefix SM20 -check-prefix SM30 %s
