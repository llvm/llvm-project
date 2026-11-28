// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=NVPTX
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -gpulibc -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=GPULIBC
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -gpulibc -O0 -emit-llvm -o - %s | FileCheck %s --check-prefix=GPULIBC
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -gpulibc -O2 -ffast-math -emit-llvm -o - %s | FileCheck %s --check-prefix=GPULIBC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -gpulibc -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_70 -nocudalib -flto -gpulibc -### -c %s 2>&1 | FileCheck %s --check-prefix=DRIVER-LIBC
// RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_70 -nocudalib -flto -### -c %s 2>&1 | FileCheck %s --check-prefix=DRIVER-DEFAULT
// RUN: %clang --target=nvptx64-nvidia-cuda -march=sm_70 -nocudalib -flto -gpulibc -nogpulibc -### -c %s 2>&1 | FileCheck %s --check-prefix=DRIVER-DEFAULT

// DRIVER-LIBC: "-cc1"{{.*}} "-gpulibc"
// DRIVER-DEFAULT-NOT: "-gpulibc"
// DRIVER-DEFAULT: "-cc1"
// DRIVER-DEFAULT-NOT: "-gpulibc"

extern double sin(double);

// NVPTX-LABEL: define{{.*}} @ordinary_sin(
// NVPTX: call double @llvm.sin.f64(
// GPULIBC-LABEL: define{{.*}} @ordinary_sin(
// GPULIBC: call{{.*}} double @sin(
double ordinary_sin(double x) { return sin(x); }

// NVPTX-LABEL: define{{.*}} @builtin_sin(
// NVPTX: call double @llvm.sin.f64(
// GPULIBC-LABEL: define{{.*}} @builtin_sin(
// GPULIBC: call{{.*}} double @sin(
// X86-LABEL: define{{.*}} @builtin_sin(
// X86: call double @llvm.sin.f64(
double builtin_sin(double x) { return __builtin_sin(x); }

// NVPTX-LABEL: define{{.*}} @builtin_sinf(
// NVPTX: call float @llvm.sin.f32(
// GPULIBC-LABEL: define{{.*}} @builtin_sinf(
// GPULIBC: call{{.*}} float @llvm.sin.f32(
float builtin_sinf(float x) { return __builtin_sinf(x); }
