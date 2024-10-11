// RUN: %clang -target nvptx64-nvidia-cuda -march=sm_61 -stdlib -startfiles \
// RUN:   -nogpulib -nogpuinc -### %s 2>&1 | FileCheck -check-prefix=NVPTX %s
// NVPTX: clang-nvlink-wrapper{{.*}}"-lc" "-lm" "{{.*}}crt1.o"
//
// RUN: %clang -target amdgcn-amd-amdhsa -march=gfx90a -stdlib -startfiles \
// RUN:   -nogpulib -nogpuinc -### %s 2>&1 | FileCheck -check-prefix=AMDGPU %s
// AMDGPU: ld.lld{{.*}}"-lc" "-lm" "{{.*}}crt1.o"
