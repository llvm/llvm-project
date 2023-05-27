// REQUIRES: amdgpu-registered-target

// RUN:   %clang -### -fopenmp --offload-arch=gfx90a -fopenmp-runtimelib=lib-debug %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Debug %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx90a -fopenmp-runtimelib=lib-perf %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Perf %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx90a -fopenmp-runtimelib=lib %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Devel %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx90a -fopenmp-target-fast %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Default %s

// RUN:   %clang -### -fopenmp --offload-arch=gfx90a -fopenmp-runtimelib=oopsy %s -O3 2>&1 \
// RUN:   | FileCheck -check-prefixes=Error %s

// Debug: /lib-debug/libomptarget
// Perf: /lib-perf/libomptarget
// Devel: /lib/libomptarget
// Default: /lib/libomptarget
// Error: clang: error: unsupported argument
