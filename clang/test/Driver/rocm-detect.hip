// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: system-windows

// Make sure the appropriate device specific library is available.

// We don't include every target in the test directory, so just pick a valid
// target not included in the test.

// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// Should not interpret -nostdlib as disabling offload libraries.
// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NODEFAULTLIBS %s

// Test ROCm installation built by SPACK by invoke clang at %T/rocm-spack/llvm-amdgpu-*
// directory through a soft link.

// RUN: rm -rf %T/rocm-spack
// RUN: cp -r %S/Inputs/rocm-spack %T
// RUN: ln -fs %clang %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK %s

// Test SPACK installation with multiple hip and rocm-device-libs packages of the same
// ROCm release. Clang cannot determine which one to use and emit diagnostics. --hip-path
// and --rocm-device-lib-path can be used to specify them.

// RUN: cp -r %T/rocm-spack/hip-* %T/rocm-spack/hip-4.0.0-abcd
// RUN: cp -r %T/rocm-spack/rocm-device-libs-* %T/rocm-spack/rocm-device-libs-4.0.0-efgh
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MULT %s
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   --hip-path=%T/rocm-spack/hip-4.0.0-abcd \
// RUN:   --rocm-device-lib-path=%T/rocm-spack/rocm-device-libs-4.0.0-efgh/amdgcn/bitcode \
// RUN:    %s 2>&1 | FileCheck -check-prefixes=SPACK-SET %s

// Test invalid SPACK ROCm installation missing hip and rocm-device-libs packages.

// RUN: rm -rf %T/rocm-spack/hip-*
// RUN: rm -rf %T/rocm-spack/rocm-device-libs-*
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MISS %s

// GFX902-DEFAULTLIBS: error: cannot find ROCm device library for gfx902. Provide its path via --rocm-path or --rocm-device-lib-path, or pass -nogpulib to build without ROCm device library

// NODEFAULTLIBS-NOT: error: cannot find

// COMMON: "-triple" "amdgcn-amd-amdhsa"

// SPACK: ROCm installation search path (Spack 4.0.0): [[DIR:.*]]
// SPACK: ROCm installation search path: [[CLANG:.*]]
// SPACK: ROCm installation search path: [[CLANG]]/lib/clang/{{[0-9.]+}}
// SPACK: ROCm installation search path: /opt/rocm
// SPACK: InstalledDir: [[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK: "-triple" "amdgcn-amd-amdhsa"
// SPACK-SAME: "-mlink-builtin-bitcode" "[[DIR]]/rocm-device-libs-4.0.0-6wnyzz4hgl3hr7uswasnagt7j2adctbs/amdgcn/bitcode/hip.bc"
// SPACK-SAME: "-internal-isystem" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"

// SPACK-MULT: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-MULT-DAG: Expecting one SPACK package hip-4.0.0 at [[DIR]] but found more
// SPACK-MULT-DAG: Expecting one SPACK package rocm-device-libs-4.0.0 at [[DIR]] but found more
// SPACK-MULT-NOT: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK-MULT-NOT: "-mlink-builtin-bitcode" "[[DIR]]/rocm-device-libs-4.0.0-6wnyzz4hgl3hr7uswasnagt7j2adctbs/amdgcn/bitcode/hip.bc"
// SPACK-MULT-NOT: "-internal-isystem" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"

// SPACK-SET: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-SET: Found HIP installation: [[DIR]]/hip-4.0.0-abcd, version 4.0.20214-a2917cd
// SPACK-SET: "-triple" "amdgcn-amd-amdhsa"
// SPACK-SET-SAME: "-mlink-builtin-bitcode" "[[DIR]]/rocm-device-libs-4.0.0-efgh/amdgcn/bitcode/hip.bc"
// SPACK-SET-SAME: "-internal-isystem" "[[DIR]]/hip-4.0.0-abcd/include"

// SPACK-MISS: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-MISS-DAG: Expecting SPACK package hip-4.0.0 at [[DIR]] but not found
// SPACK-MISS-DAG: Expecting SPACK package rocm-device-libs-4.0.0 at [[DIR]] but not found
// SPACK-MISS-NOT: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK-MISS-NOT: "-mlink-builtin-bitcode" "[[DIR]]/rocm-device-libs-4.0.0-6wnyzz4hgl3hr7uswasnagt7j2adctbs/amdgcn/bitcode/hip.bc"
// SPACK-MISS-NOT: "-internal-isystem" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"
