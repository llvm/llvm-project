// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

#ifdef __HIP_DEVICE_COMPILE__
DEVICE __SSE3__
#else
HOST __SSE3__
#endif

// RUN: %clang -x hip -E -target x86_64-linux-gnu -msse3 --cuda-gpu-arch=gfx803 -nogpulib -nogpuinc -nobuiltininc -o - %s 2>&1 | FileCheck %s

// CHECK-NOT: SSE3
