// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
//
// Check that we properly detect CUDA installation.
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/no-cuda-there --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=i386-apple-macosx \
// RUN:   --sysroot=%S/no-cuda-there --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=x86_64-unknown-linux \
// RUN:   --sysroot=%S/no-cuda-there --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=x86_64-apple-macosx \
// RUN:   --sysroot=%S/no-cuda-there --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA


// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/Inputs/CUDA --cuda-path-ignore-env 2>&1 | FileCheck %s
// RUN: %clang -v --target=i386-apple-macosx \
// RUN:   --sysroot=%S/Inputs/CUDA --cuda-path-ignore-env 2>&1 | FileCheck %s

// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s
// RUN: %clang -v --target=i386-apple-macosx \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda 2>&1 | FileCheck %s

// Check that we don't find a CUDA installation without libdevice ...
// RUN: %clang -v --target=i386-unknown-linux \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=i386-apple-macosx \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=x86_64-unknown-linux \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA
// RUN: %clang -v --target=x84_64-apple-macosx \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NOCUDA

// ... unless the user doesn't need libdevice
// RUN: %clang -v --target=i386-unknown-linux -nocudalib \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NO-LIBDEVICE
// RUN: %clang -v --target=i386-apple-macosx -nocudalib \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NO-LIBDEVICE
// RUN: %clang -v --target=x86_64-unknown-linux -nocudalib \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NO-LIBDEVICE
// RUN: %clang -v --target=x86_64-apple-macosx -nocudalib \
// RUN:   --sysroot=%S/Inputs/CUDA-nolibdevice --cuda-path-ignore-env 2>&1 | FileCheck %s -check-prefix NO-LIBDEVICE


// Make sure we map libdevice bitcode files to proper GPUs. These
// tests use Inputs/CUDA_80 which has full set of libdevice files.
// However, libdevice mapping only matches CUDA-7.x at the moment.
// sm_2x, sm_32 -> compute_20
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_21 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE20
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_32 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE20
// sm_30, sm_6x map to compute_30.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_30 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE30
// sm_5x is a special case. Maps to compute_30 for cuda-7.x only.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_50 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE30
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_60 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE30
// sm_35 and sm_37 -> compute_35
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix CUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_37 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix CUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35
// sm_5x -> compute_50 for CUDA-8.0 and newer.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_50 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE50

// Verify that -nocudainc prevents adding include path to CUDA headers.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   -nocudainc --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35
// RUN: %clang -### -v --target=i386-apple-macosx --cuda-gpu-arch=sm_35 \
// RUN:   -nocudainc --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC \
// RUN:     -check-prefix LIBDEVICE -check-prefix LIBDEVICE35

// We should not add any CUDA include paths if there's no valid CUDA installation
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC
// RUN: %clang -### -v --target=i386-apple-macosx --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOCUDAINC

// Verify that we get an error if there's no libdevice library to link with.
// NOTE: Inputs/CUDA deliberately does *not* have libdevice.compute_20  for this purpose.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_20 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix MISSINGLIBDEVICE
// RUN: %clang -### -v --target=i386-apple-macosx --cuda-gpu-arch=sm_20 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix MISSINGLIBDEVICE

// Verify that  -nocudalib prevents linking libdevice bitcode in.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOLIBDEVICE
// RUN: %clang -### -v --target=i386-apple-macosx --cuda-gpu-arch=sm_35 \
// RUN:   -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON -check-prefix NOLIBDEVICE

// Verify that we don't add include paths, link with libdevice or
// -include __clang_cuda_runtime_wrapper.h without valid CUDA installation.
// RUN: %clang -### -v --target=i386-unknown-linux --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix NOCUDAINC -check-prefix NOLIBDEVICE
// RUN: %clang -### -v --target=i386-apple-macosx --cuda-gpu-arch=sm_35 \
// RUN:   --cuda-path=%S/no-cuda-there %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix COMMON \
// RUN:     -check-prefix NOCUDAINC -check-prefix NOLIBDEVICE

// Verify that C++ include paths are passed for both host and device frontends.
// RUN: %clang -### -no-canonical-prefixes -target x86_64-linux-gnu %s \
// RUN: --stdlib=libstdc++ --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree2 \
// RUN: --gcc-toolchain="" 2>&1 \
// RUN: | FileCheck %s --check-prefix CHECK-CXXINCLUDE

// Verify that CUDA SDK version is propagated to the CC1 compilations.
// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=sm_50 \
// RUN:   --cuda-path=%S/Inputs/CUDA_80/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix CUDA80

// Verify that if no version file is found, we report the default of 7.0.
// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=sm_50 \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix CUDA70

// CHECK: Found CUDA installation: {{.*}}/Inputs/CUDA/usr/local/cuda
// NO-LIBDEVICE: Found CUDA installation: {{.*}}/Inputs/CUDA-nolibdevice/usr/local/cuda
// NOCUDA-NOT: Found CUDA installation:

// MISSINGLIBDEVICE: error: cannot find libdevice for sm_20.

// COMMON: "-triple" "nvptx-nvidia-cuda"
// COMMON-SAME: "-fcuda-is-device"
// LIBDEVICE-SAME: "-mlink-builtin-bitcode"
// NOLIBDEVICE-NOT: "-mlink-builtin-bitcode"
// LIBDEVICE20-SAME: libdevice.compute_20.10.bc
// LIBDEVICE30-SAME: libdevice.compute_30.10.bc
// LIBDEVICE35-SAME: libdevice.compute_35.10.bc
// LIBDEVICE50-SAME: libdevice.compute_50.10.bc
// NOLIBDEVICE-NOT: libdevice.compute_{{.*}}.bc
// LIBDEVICE-SAME: "-target-feature" "+ptx42"
// NOLIBDEVICE-NOT: "-target-feature" "+ptx42"
// CUDAINC-SAME: "-internal-isystem" "{{.*}}/Inputs/CUDA{{[_0-9]+}}/usr/local/cuda/include"
// NOCUDAINC-NOT: "-internal-isystem" "{{.*}}/cuda/include"
// CUDAINC-SAME: "-include" "__clang_cuda_runtime_wrapper.h"
// NOCUDAINC-NOT: "-include" "__clang_cuda_runtime_wrapper.h"
// -internal-externc-isystem flags must come *after* the cuda include flags,
// because we must search the cuda include directory first.
// CUDAINC-SAME: "-internal-externc-isystem"
// COMMON-SAME: "-x" "cuda"
// CHECK-CXXINCLUDE: clang{{.*}} "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CHECK-CXXINCLUDE-SAME: {{.*}}"-internal-isystem" "{{.+}}/include/c++/4.8"
// CHECK-CXXINCLUDE: clang{{.*}} "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CHECK-CXXINCLUDE-SAME: {{.*}}"-internal-isystem" "{{.+}}/include/c++/4.8"
// CHECK-CXXINCLUDE: ld{{.*}}"

// CUDA80: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA80-SAME: -target-sdk-version=8.0
// CUDA80: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CUDA80-SAME: -target-sdk-version=8.0
// CUDA80: ld{{.*}}"

// CUDA70: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CUDA70-SAME: -target-sdk-version=7.0
// CUDA70: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CUDA70-SAME: -target-sdk-version=7.0
// CUDA70: ld{{.*}}"
