// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
//
// # Check that we properly detect CUDA installation.
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/no-cuda-there 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/Inputs/CUDA 2>&1 | FileCheck %s
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s

// Make sure we map libdevice bitcode files to proper GPUs.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_21 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE21
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix CUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35
// Verify that -nocudainc prevents adding include path to CUDA headers.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   -nocudainc --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35
// We should not add any CUDA include paths if there's no valid CUDA installation
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC

// Verify that no options related to bitcode linking are passes if
// there's no bitcode file.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_30 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOLIBDEVICE
// .. or if we explicitly passed -nocudalib
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOLIBDEVICE
// Verify that we don't add include paths, link with libdevice or
// -include __clang_cuda_runtime_wrapper.h without valid CUDA installation.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix NOCUDAINC -check-prefix NOLIBDEVICE

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// NOCUDA-NOT: Found CUDA installation:

// COMMON: "-triple" "nvptx-nvidia-cuda"
// COMMON-SAME: "-fcuda-is-device"
// LIBDEVICE-SAME: "-mlink-cuda-bitcode"
// NOLIBDEVICE-NOT: "-mlink-cuda-bitcode"
// LIBDEVICE21-SAME: libdevice.compute_20.10.bc
// LIBDEVICE35-SAME: libdevice.compute_35.10.bc
// NOLIBDEVICE-NOT: libdevice.compute_{{.*}}.bc
// LIBDEVICE-SAME: "-target-feature" "+ptx42"
// NOLIBDEVICE-NOT: "-target-feature" "+ptx42"
// CUDAINC-SAME: "-internal-isystem" "{{.*}}/Inputs/CUDA/usr/local/cuda/include"
// NOCUDAINC-NOT: "-internal-isystem" "{{.*}}/cuda/include"
// CUDAINC-SAME: "-include" "__clang_cuda_runtime_wrapper.h"
// NOCUDAINC-NOT: "-include" "__clang_cuda_runtime_wrapper.h"
// COMMON-SAME: "-x" "cuda"
