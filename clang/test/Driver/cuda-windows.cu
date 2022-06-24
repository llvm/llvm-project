// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -v --target=i386-pc-windows-msvc \
// RUN:   --sysroot=%S/Inputs/CUDA-windows 2>&1 %s -### | FileCheck %s
// RUN: %clang -v --target=i386-pc-windows-mingw32 \
// RUN:   --sysroot=%S/Inputs/CUDA-windows 2>&1 %s -### | FileCheck %s

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA-windows/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0
// CHECK: "-cc1" "-triple" "nvptx-nvidia-cuda"
// CHECK-SAME: "-fms-extensions"
// CHECK-SAME: "-fms-compatibility"
// CHECK-SAME: "-fms-compatibility-version=
