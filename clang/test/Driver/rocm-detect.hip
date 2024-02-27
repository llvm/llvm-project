// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: system-windows

// Make sure the appropriate device specific library is available.

// We don't include every target in the test directory, so just pick a valid
// target not included in the test.

// RUN: not %clang -### -v --target=x86_64-linux-gnu --cuda-gpu-arch=gfx902 \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// Should not interpret -nostdlib as disabling offload libraries.
// RUN: not %clang -### -v --target=x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nostdlib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,GFX902-DEFAULTLIBS %s

// RUN: %clang -### -v -target x86_64-linux-gnu --cuda-gpu-arch=gfx902 -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=COMMON,NODEFAULTLIBS %s

// Test environment variable ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm %clang -### --target=x86_64-linux-gnu \
// RUN:   --print-rocm-search-dirs --offload-arch=gfx1010 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV %s

// Test interaction between environment variables HIP_PATH and ROCM_PATH.
// Device libs are found under ROCM_PATH. HIP include files and HIP runtime library
// are found under HIP_PATH.

// RUN: rm -rf %t/myhip
// RUN: mkdir -p %t/myhip
// RUN: cp -r %S/Inputs/rocm/bin %t/myhip

// Test HIP_PATH overrides ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm HIP_PATH=%t/myhip \
// RUN:   %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV,HIP-PATH %s

// Test --hip-path overrides ROCM_PATH.
// RUN: env ROCM_PATH=%S/Inputs/rocm \
// RUN:   %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --hip-path=%t/myhip \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV,HIP-PATH %s

// Test --hip-path overrides --rocm-path.
// RUN: %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --hip-path=%t/myhip --rocm-path=%S/Inputs/rocm \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV,HIP-PATH %s

// Test HIP_PATH overrides --rocm-path.
// RUN: env HIP_PATH=%t/myhip %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --rocm-path=%S/Inputs/rocm \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV,HIP-PATH %s

// Test empty HIP_PATH does not override --rocm-path.
// RUN: env HIP_PATH= \
// RUN:   %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   --rocm-path=%S/Inputs/rocm --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-PATH %s

// Test --hip-path option overrides environment variable HIP_PATH.

// RUN: rm -rf %t/myhip
// RUN: rm -rf %t/myhip_nouse
// RUN: mkdir -p %t/myhip
// RUN: mkdir -p %t/myhip_nouse
// RUN: cp -r %S/Inputs/rocm/bin %t/myhip
// RUN: cp -r %S/Inputs/rocm/bin %t/myhip_nouse
// RUN: env ROCM_PATH=%S/Inputs/rocm HIP_PATH=%t/myhip_nouse \
// RUN:   %clang -### -target x86_64-linux-gnu --offload-arch=gfx1010 --hip-link \
// RUN:   --hip-path=%t/myhip --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-ENV,HIP-PATH %s

// Test detecting /usr directory.
// RUN: rm -rf %t/*
// RUN: cp -r %S/Inputs/rocm %t/usr
// RUN: mkdir -p %t/usr/share/hip
// RUN: mv %t/usr/bin/.hipVersion %t/usr/share/hip/version
// RUN: mkdir -p %t/usr/local
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx1010 --sysroot=%t \
// RUN:   --print-rocm-search-dirs --hip-link %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=USR %s

// Test detecting latest /opt/rocm-{release} directory.
// RUN: rm -rf %t/*
// RUN: mkdir -p %t/opt
// RUN: cp -r %S/Inputs/rocm %t/opt/rocm-3.9.0-1234
// RUN: cp -r %S/Inputs/rocm %t/opt/rocm-3.10.0
// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx1010 --sysroot=%t \
// RUN:   --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=ROCM-REL %s

// Test ROCm installation built by SPACK by invoke clang at %t/rocm-spack/llvm-amdgpu-*
// directory through a soft link.

// RUN: rm -rf %t/rocm-spack
// RUN: cp -r %S/Inputs/rocm-spack %t
// RUN: ln -fs %clang %t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang
// RUN: %t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -resource-dir=%t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/lib/clang \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 --print-rocm-search-dirs %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK %s

// Test SPACK installation with multiple hip and rocm-device-libs packages of the same
// ROCm release. --hip-path and --rocm-device-lib-path can be used to specify them.

// RUN: cp -r %t/rocm-spack/hip-* %t/rocm-spack/hip-4.0.0-abcd
// RUN: %t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang -### -v \
// RUN:   -target x86_64-linux-gnu --cuda-gpu-arch=gfx900 \
// RUN:   --hip-path=%t/rocm-spack/hip-4.0.0-abcd \
// RUN:    %s 2>&1 | FileCheck -check-prefixes=SPACK-SET %s

// Test invalid SPACK ROCm installation missing hip and rocm-device-libs packages.

// RUN: rm -rf %t/rocm-spack/hip-*
// RUN: rm -rf %t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/amdgcn
// RUN: %t/rocm-spack/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin/clang --version 2>&1 \
// RUN:   | FileCheck -check-prefixes=SPACK-MISS-SILENT %s

// GFX902-DEFAULTLIBS: error: cannot find ROCm device library for gfx902; provide its path via '--rocm-path' or '--rocm-device-lib-path', or pass '-nogpulib' to build without ROCm device library

// NODEFAULTLIBS-NOT: error: cannot find

// COMMON: "-triple" "amdgcn-amd-amdhsa"

// ROCM-ENV: ROCm installation search path: [[ROCM_PATH:.*/Inputs/rocm]]

// HIP-PATH: "-mlink-builtin-bitcode" "[[ROCM_PATH]]/amdgcn/bitcode/oclc_isa_version_1010.bc"
// HIP-PATH: "-idirafter" "[[HIP_PATH:.*/myhip]]/include"
// HIP-PATH: "-L[[HIP_PATH]]/lib" {{.*}}"-lamdhip64"

// ROCM-PATH: ROCm installation search path: [[ROCM_PATH:.*/Inputs/rocm]]
// ROCM-PATH: "-mlink-builtin-bitcode" "[[ROCM_PATH]]/amdgcn/bitcode/oclc_isa_version_1010.bc"
// ROCM-PATH: "-idirafter" "[[ROCM_PATH]]/include"
// ROCM-PATH: "-L[[ROCM_PATH]]/lib" {{.*}}"-lamdhip64"

// USR: ROCm installation search path: [[ROCM_PATH:.*/usr$]]
// USR: "-mlink-builtin-bitcode" "[[ROCM_PATH]]/amdgcn/bitcode/oclc_isa_version_1010.bc"
// USR: "-idirafter" "[[ROCM_PATH]]/include"
// USR: "-L[[ROCM_PATH]]/lib" {{.*}}"-lamdhip64"

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

// SPACK-SET: InstalledDir: [[DIR:.*]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/bin
// SPACK-SET: Found HIP installation: [[DIR]]/hip-4.0.0-abcd, version 4.0.20214-a2917cd
// SPACK-SET: "-triple" "amdgcn-amd-amdhsa"
// SPACK-SET-SAME: "-mlink-builtin-bitcode" "[[DIR]]/llvm-amdgpu-4.0.0-ieagcs7inf7runpyfvepqkurasoglq4z/amdgcn/bitcode/hip.bc"
// SPACK-SET-SAME: "-idirafter" "[[DIR]]/hip-4.0.0-abcd/include"

// SPACK-MISS-SILENT-NOT: SPACK package hip-{{.*}} not found at
// SPACK-MISS-SILENT-NOT: Found HIP installation
