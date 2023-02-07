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

// Test environment variable ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm %clang -### -target x86_64-linux-gnu \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV %s

// Test detecting latest /opt/rocm-{release} directory.
// RUN: rm -rf %T/opt
// RUN: mkdir -p %T/opt
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.9.0-1234
// RUN: cp -r %S/Inputs/rocm %T/opt/rocm-3.10.0
// RUN: %clang -### -target x86_64-linux-gnu --sysroot=%T \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-REL %s

// Test ROCm installation built by SPACK by invoke clang at %T/rocm-spack/llvm-amdgpu-*
// directory through a soft link.

// RUN: rm -rf %T/rocm-spack
// RUN: cp -r %S/Inputs/rocm-spack %T
// RUN: ln -fs %clang %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -resource-dir=%T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/lib/clang \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK %s

// Test SPACK installation with multiple hip and rocm-device-libs packages of the same
// ROCm release. Clang cannot determine which one to use and emit diagnostics. --hip-path
// and --rocm-device-lib-path can be used to specify them.

// RUN: cp -r %T/rocm-spack/hip-* %T/rocm-spack/hip-4.0.0-abcd
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MULT %s
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   --hip-path=%T/rocm-spack/hip-4.0.0-abcd \
// RUN:    %s 2>&1 | FileCheck -check-prefixes=SPACK-SET %s

// Test invalid SPACK ROCm installation missing hip and rocm-device-libs packages.
// The message about SPACK is emitted only if -v is specified.

// RUN: rm -rf %T/rocm-spack/hip-*
// RUN: rm -rf %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/amdgcn
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MISS %s
// RUN: %T/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang --version 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MISS-SILENT %s

// GFX902-DEFAULTLIBS: error: cannot find ROCm device library for gfx902; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library

// NODEFAULTLIBS-NOT: error: cannot find

// COMMON: "-triple" "amdgcn-amd-amdhsa"

// ROCM-ENV: ROCm installation search path: {{.*}}/Inputs/rocm

// ROCM-REL: ROCm installation search path: {{.*}}/opt/rocm
// ROCM-REL: ROCm installation search path: {{.*}}/opt/rocm-3.10.0

// SPACK: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK: ROCm installation search path (Spack 4.0.0): [[DIR]]
// SPACK: ROCm installation search path: [[CLANG:.*]]
// SPACK: ROCm installation search path: [[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z
// SPACK: ROCm installation search path: [[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/lib/clang
// SPACK: ROCm installation search path: /opt/rocm
// SPACK: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK: "-triple" "amdgcn-amd-amdhsa"
// SPACK-SAME: "-mlink-builtin-bitcode" "[[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/amdgcn/bitcode/hip.bc"
// SPACK-SAME: "-idirafter" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"

// SPACK-MULT: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-MULT-DAG: Cannot use SPACK package hip-4.0.0 at [[DIR]] due to multiple installations for the same version
// SPACK-MULT-NOT: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK-MULT-NOT: "-internal-isystem" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"

// SPACK-SET: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-SET: Found HIP installation: [[DIR]]/hip-4.0.0-abcd, version 4.0.20214-a2917cd
// SPACK-SET: "-triple" "amdgcn-amd-amdhsa"
// SPACK-SET-SAME: "-mlink-builtin-bitcode" "[[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/amdgcn/bitcode/hip.bc"
// SPACK-SET-SAME: "-idirafter" "[[DIR]]/hip-4.0.0-abcd/include"

// SPACK-MISS: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-MISS-DAG: SPACK package hip-4.0.0 not found at [[DIR]]
// SPACK-MISS-NOT: Found HIP installation: [[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5, version 4.0.20214-a2917cd
// SPACK-MISS-NOT: "-idirafter" "[[DIR]]/hip-4.0.0-5f63slrursbrvfe2txrrjkynbsywsob5/include"

// SPACK-MISS-SILENT-NOT: SPACK package hip-{{.*}} not found at
// SPACK-MISS-SILENT-NOT: Found HIP installation
