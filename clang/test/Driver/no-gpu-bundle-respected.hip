// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=OFFLOAD

// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --gpu-bundle-output --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=OFFLOAD

// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --no-gpu-bundle-output --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=OFFLOAD2

// OFFLOAD: clang-offload-bundler
// OFFLOAD2-NOT: clang-offload-bundler

int square(int num) {
    return num * num;
}
