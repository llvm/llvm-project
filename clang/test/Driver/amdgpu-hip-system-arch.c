// REQUIRES: shell
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_fail %t/
// RUN: cp %S/Inputs/amdgpu-arch/amdgpu_arch_gfx906 %t/
// RUN: echo '#!/bin/sh' > %t/amdgpu_arch_empty
// RUN: chmod +x %t/amdgpu_arch_fail
// RUN: chmod +x %t/amdgpu_arch_gfx906
// RUN: chmod +x %t/amdgpu_arch_empty

// case when amdgpu-arch returns nothing or fails
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_fail -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_fail -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NO-OUTPUT-ERROR
// NO-OUTPUT-ERROR: error: cannot determine hip architecture{{.*}}; consider passing it via '--offload-arch'

// case when amdgpu-arch does not return anything with successful execution
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_empty -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_empty -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=EMPTY-OUTPUT
// EMPTY-OUTPUT: error: cannot determine hip architecture: No GPU detected in the system; consider passing it via '--offload-arch'

// case when amdgpu-arch returns a gfx906 GPU.
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-GFX906
// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib --offload-new-driver --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 -x hip %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ARCH-GFX906
// ARCH-GFX906: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"

// case when CLANG_TOOLCHAIN_PROGRAM_TIMEOUT is malformed.
// RUN: env CLANG_TOOLCHAIN_PROGRAM_TIMEOUT=foo \
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -nogpuinc -nogpulib \
// RUN:     --offload-arch=native --amdgpu-arch-tool=%t/amdgpu_arch_gfx906 \
// RUN:     -x hip %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=BAD-TIMEOUT
// BAD-TIMEOUT: clang: error: cannot determine hip architecture: CLANG_TOOLCHAIN_PROGRAM_TIMEOUT expected an integer, got 'foo'; consider passing it via '--offload-arch'; environment variable CLANG_TOOLCHAIN_PROGRAM_TIMEOUT specifies the tool timeout (integer secs, <=0 is infinite)
