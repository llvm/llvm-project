// Check that -mcpu with a mismatched subarch from the triple are rejected
// RUN: not %clang -### --target=amdgpu7-amd-amdhsa -mcpu=gfx600 -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR0 %s
// RUN: not %clang -### --target=amdgpu6.00-amd-amdhsa -mcpu=gfx601 -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR1 %s
// RUN: not %clang -### --target=amdgpu11-amd-amdhsa -mcpu=gfx900 -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR2 %s
// RUN: not %clang -### --target=amdgpu11-amd-amdhsa -mcpu=gfx900:xnack+ -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR2 %s
// RUN: not %clang -### --target=amdgpu12.50-amd-amdhsa -mcpu=gfx900 -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR3 %s
// RUN: not %clang -### --target=amdgpu12.50-amd-amdhsa -march=gfx900 -nogpulib -c %s 2>&1 | FileCheck -check-prefix=ERR3 %s

// ERR0: error: the target architecture 'gfx600' is not supported by the target 'amdgpu7'
// ERR1: error: the target architecture 'gfx601' is not supported by the target 'amdgpu6.00'
// ERR2: error: the target architecture 'gfx900' is not supported by the target 'amdgpu11'
// ERR3: error: the target architecture 'gfx900' is not supported by the target 'amdgpu12.50'
