// RUN: not %clang -### --target=x86_64-pc-linux-gnu --offload-targets=amdgpu9-amd-amdhsa --offload-arch=gfx1030 -nogpulib -nogpuinc -c %s 2>&1 | FileCheck -check-prefix=ERR0 %s

// RUN: not %clang -### --target=x86_64-pc-linux-gnu --offload-targets=amdgpu9.00-amd-amdhsa --offload-arch=gfx902 -nogpulib -nogpuinc -c %s 2>&1 | FileCheck -check-prefix=ERR1 %s

// RUN: %clang -### --target=x86_64-pc-linux-gnu --offload-targets=amdgpu9-amd-amdhsa --offload-arch=gfx900 -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --allow-empty -check-prefix=NO-ERR %s
// RUN: %clang -### --target=x86_64-pc-linux-gnu --offload-targets=amdgpu9-amd-amdhsa -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --allow-empty -check-prefix=NO-ERR %s
// RUN: %clang -### --target=x86_64-pc-linux-gnu --offload-targets=amdgpu9-amd-amdhsa --offload-arch=gfx902 -nogpulib -nogpuinc -c %s 2>&1 | FileCheck --allow-empty -check-prefix=NO-ERR %s


// ERR0: error: the target architecture 'gfx1030' is not supported by the target 'amdgpu9'
// ERR1: error: the target architecture 'gfx902' is not supported by the target 'amdgpu9.00'

// NO-ERR: {{^$}}
