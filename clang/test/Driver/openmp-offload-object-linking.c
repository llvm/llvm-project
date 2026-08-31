// REQUIRES: amdgpu-registered-target

// OpenMP supports full LTO and ThinLTO, but not explicit object-linking
// options.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -foffload-object-linking \
// RUN:   -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ENABLE
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -foffload-lto=thin \
// RUN:   -foffload-object-linking -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ENABLE
// ENABLE: error: unsupported option '-foffload-object-linking' for language mode 'OpenMP'

// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fopenmp \
// RUN:   --offload-arch=gfx906 -foffload-lto=thin \
// RUN:   -fno-offload-object-linking -nogpulib -nogpuinc %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLE
// DISABLE: error: unsupported option '-fno-offload-object-linking' for language mode 'OpenMP'

int main(void) { return 0; }
