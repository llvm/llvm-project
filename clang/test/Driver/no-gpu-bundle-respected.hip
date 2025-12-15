// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=BUNDLE

// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --gpu-bundle-output --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=BUNDLE

// RUN: %clang -ccc-print-phases -c -emit-llvm \
// RUN:   --no-gpu-bundle-output --offload-arch=gfx900,gfx1030 -O3 -x hip %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=COMPILER,GFX1030,GFX900,OFFLOAD,NOBUNDLE

// BUNDLE: clang-offload-bundler
// NOBUNDLE-NOT: clang-offload-bundler

// COM: sanity checks
// COMPILER: compiler
// GFX1030: (device-hip, gfx1030)
// GFX900: (device-hip, gfx900)
// OFFLOAD: offload

int square(int num) {
    return num * num;
}
