// REQUIRES: zstd

// Test HIP offload compression.

// RUN: rm -rf %t.bc
// RUN: %clang -### -v --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress --offload-compression-level=9 \
// RUN:   --gpu-bundle-output \
// RUN:   -o %t.bc \
// RUN: 2>&1 | FileCheck %s --check-prefix=NEWDRIVER

// NEWDRIVER: llvm-offload-binary{{.*}}"--image={{[^"]*}}arch=gfx1101{{[^"]*}}" "--compress" "--compression-format=zstd" "--compression-level=9"{{$}}
// NEWDRIVER: clang-linker-wrapper{{.*}}"--compress" "--compression-level=9"
// NEWDRIVER-NOT: "--verbose"

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   -nogpuinc -nogpulib -c %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress \
// RUN: 2>&1 | FileCheck %s --check-prefix=NORDC

// NORDC: llvm-offload-binary{{.*}}"--image={{[^"]*}}arch=gfx1101{{[^"]*}}"{{$}}
// NORDC-NOT: "--compress"
// NORDC: clang-linker-wrapper{{.*}}"--emit-fatbin-only"{{.*}}"--compress"

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 \
// RUN:   -fgpu-rdc -nogpuinc -nogpulib -c %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress --no-offload-compress \
// RUN: 2>&1 | FileCheck %s --check-prefix=NOCOMPRESS

// NOCOMPRESS: llvm-offload-binary{{.*}}"--image={{[^"]*}}arch=gfx1100{{[^"]*}}"{{$}}

// --offload-compress is HIP only, so other offload packagers ignore it.
// RUN: %clang -### --target=x86_64-linux-gnu -x c -fopenmp \
// RUN:   --offload-arch=gfx1100 -nogpulib --offload-compress -c %s \
// RUN: 2>&1 | FileCheck %s --check-prefix=OMP

// OMP: llvm-offload-binary{{.*}}kind=openmp
// OMP-NOT: "--compress"
