// REQUIRES: x86-registered-target, amdgpu-registered-target

// Not passed by default.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s

// Passed through to -cc1 when requested.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=ENABLE %s

// Explicit disable wins over a preceding enable and is not passed through.
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -fopenmp-target-fast-reduction -fno-openmp-target-fast-reduction %s 2>&1 \
// RUN:   | FileCheck -check-prefix=DEFAULT %s

// DEFAULT-NOT: {{"-f(no-)?openmp-target-fast-reduction"}}

// ENABLE: "-fopenmp-target-fast-reduction"
// ENABLE-NOT: "-fno-openmp-target-fast-reduction"
