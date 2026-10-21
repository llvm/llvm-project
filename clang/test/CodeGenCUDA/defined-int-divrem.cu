// CUDA device code defaults to defined integer division and remainder; host
// code retains undefined behavior. An explicit opt-out restores plain ops.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fno-defined-int-divrem -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))

__host__ __device__ int f_sdiv(int a, int b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_sdiv
// ON: = call i32 @llvm.sdiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_sdiv
// OFF: = sdiv i32

__host__ __device__ int f_srem(int a, int b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_srem
// ON: = call i32 @llvm.srem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_srem
// OFF: = srem i32

__host__ __device__ unsigned f_udiv(unsigned a, unsigned b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_udiv
// ON: = call i32 @llvm.udiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_udiv
// OFF: = udiv i32

__host__ __device__ unsigned f_urem(unsigned a, unsigned b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_urem
// ON: = call i32 @llvm.urem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_urem
// OFF: = urem i32

// A safe constant divisor keeps the plain instruction in both modes.
__host__ __device__ int f_divc(int a) { return a / 4; }
// ON-LABEL: define {{.*}} @{{.*}}f_divc
// ON: = sdiv i32 {{.*}}, 4
// OFF-LABEL: define {{.*}} @{{.*}}f_divc
// OFF: = sdiv i32 {{.*}}, 4
