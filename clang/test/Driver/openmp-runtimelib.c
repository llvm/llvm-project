// REQUIRES: amdgpu-registered-target

// RUN:  %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib-debug %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Debug,Debug-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib-perf %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Perf,Perf-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=lib %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Devel,Devel-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-target-fast %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Devel,Devel-Rel %s

// RUN: not %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a -fopenmp-runtimelib=oopsy %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Error %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib-debug -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Asan-Debug,Asan-Debug-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Asan-Devel,Asan-Devel-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-runtimelib=lib-perf -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Asan-Perf,Asan-Perf-Rel %s

// RUN: %clang -### -fopenmp -nogpuinc -nogpulib  --offload-arch=gfx90a:xnack+ -fopenmp-target-fast -fsanitize=address -shared-libasan %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Asan-Devel,Asan-Devel-Rel %s

// Devel: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib]]"
// Devel-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Debug: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib-debug]]"
// Debug-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Perf: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib-perf]]"
// Perf-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Asan-Devel: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib(/|\\\\)asan]]"
// Asan-Devel-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Asan-Debug: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib-debug(/|\\\\)asan]]"
// Asan-Debug-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Asan-Perf: "-rpath" "{{[^"]*}}[[LIB:(/|\\\\)lib-perf(/|\\\\)asan]]"
// Asan-Perf-Rel-NOT: "-rpath" "{{[^"]*(/|\\\\)\.\.}}[[LIB]]"

// Error: clang: error: unsupported argument 'oopsy' to option '-fopenmp-runtimelib='
