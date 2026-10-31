// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib %s -O0 2>&1 \
// RUN:   | FileCheck -check-prefixes=DefaultTFast,DefaultTState,DefaultNoNestParallel,DefaultTeamsOver,DefaultThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O0 -fopenmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TState,NestParallel,TeamsOver,ThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O3 %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O3,DefaultTFast,DefaultTState,DefaultNoNestParallel,DefaultTeamsOver,DefaultThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O3 -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=O3,DefaultTState,DefaultNoNestParallel,DefaultTeamsOver,DefaultThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -Ofast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,TState,NestParallel,TeamsOver,ThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -Ofast -fno-openmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=OFast,DefaultTState,DefaultNoNestParallel,DefaultTeamsOver,DefaultThreadsOver %s

// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a -nogpulib -O0 -fno-openmp-target-fast -fopenmp-target-fast %s 2>&1 \
// RUN:   | FileCheck -check-prefixes=TState,NestParallel,TeamsOver,ThreadsOver %s

// O3: -O3
// OFast: -Ofast

// DefaultTFast-NOT: {{"-f(no-)?openmp-target-fast"}}

// TState: "-fopenmp-assume-no-thread-state"
// TState-NOT: "-fno-openmp-assume-no-thread-state"
// DefaultTState-NOT: {{"-f(no-)?openmp-assume-no-thread-state"}}

// NestParallel: "-fopenmp-assume-no-nested-parallelism"
// NestParallel-NOT: "-fno-openmp-assume-no-nested-parallelism"
// DefaultNoNestParallel-NOT: {{"-f(-no-)?openmp-assume-no-nested-parallelism"}}

// TeamsOver: "-fopenmp-assume-teams-oversubscription"
// TeamsOver-NOT: "-fno-openmp-assume-teams-oversubscription"
// DefaultTeamsOver-NOT: {{"-f(no-)?openmp-assume-teams-oversubscription"}}

// ThreadsOver: "-fopenmp-assume-threads-oversubscription"
// ThreadsOver-NOT: "-fno-openmp-assume-threads-oversubscription"
// DefaultThreadsOver-NOT: {{"-f(no-)?openmp-assume-threads-oversubscription"}}
