// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a %s -O0 2>&1 \
// RUN:   | FileCheck -check-prefixes=NoTFast,NoEnV,NoTState,NoNestParallel %s

// RUN:  %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -O0 -fopenmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TFast,EnV,TState,NestParallel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -O4 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O4,NoTFast,NoEnV,NoTState,NoNestParallel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -O4 -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O4,NoTFast,NoEnV,NoTState,NoNestParallel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -Ofast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,TFast,EnV,TState,NestParallel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -Ofast -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,NoTFast,NoEnV,NoTState,NoNestParallel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -fopenmp-target-fast -fno-openmp-target-ignore-env-vars %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TFast,NoEnV,TState,NestParallel,O3 %s

// O4: -O4
// OFast: -Ofast
// O3: -O3

// TFast: -fopenmp-target-fast
// TFast-NOT: -fno-openmp-target-fast
// NoTFast: -fno-openmp-target-fast
// NoTFast-NOT: -fopenmp-target-fast

// EnV: -fopenmp-target-ignore-env-vars
// EnV-NOT: -fno-openmp-target-ignore-env-vars
// NoEnV: -fno-openmp-target-ignore-env-vars
// NoEnV-NOT: -fopenmp-target-ignore-env-vars

// TState: -fopenmp-assume-no-thread-state
// TState-NOT: -fno-openmp-assume-no-thread-state
// NoTState: -fno-openmp-assume-no-thread-state
// NoTState-NOT: -fopenmp-assume-no-thread-state

// NestParallel: -fopenmp-assume-no-nested-parallelism
// NestParallel-NOT: -fno-openmp-assume-no-nested-parallelism
// NoNestParallel: -fno-openmp-assume-no-nested-parallelism
// NoNestParallel-NOT: -fopenmp-assume-no-nested-parallelism
