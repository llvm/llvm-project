// REQUIRES: zstd
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Test compress bundled bitcode.

// RUN: rm -rf %t.bc
// RUN: %clang -c -v --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   --no-offload-new-driver -fgpu-rdc -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress --offload-compression-level=9 \
// RUN:   --offload-device-only --gpu-bundle-output \
// RUN:   -o %t.bc \
// RUN: 2>&1 | FileCheck %s

// CHECK: clang-offload-bundler{{.*}} -type=bc
// CHECK-SAME: -targets={{.*}}hip-amdgcn-amd-amdhsa-gfx1100,hip-amdgcn-amd-amdhsa-gfx1101
// CHECK-SAME: -compress -verbose -compression-level=9
// CHECK: Compressed bundle format

// Test uncompress of bundled bitcode.

// RUN: %clang --hip-link -### -v --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   --no-offload-new-driver -fgpu-rdc -nogpulib \
// RUN:   %t.bc --offload-device-only \
// RUN: 2>&1 | FileCheck -check-prefix=UNBUNDLE %s

// UNBUNDLE: clang-offload-bundler{{.*}} "-type=bc"
// UNBUNDLE-SAME: -targets={{.*}}hip-amdgcn-amd-amdhsa-gfx1100,hip-amdgcn-amd-amdhsa-gfx1101
// UNBUNDLE-SAME: -unbundle
// UNBUNDLE-SAME: -verbose

// Test compress bundled code objects.

// RUN: %clang -c -### -v --target=x86_64-linux-gnu \
// RUN:   -x hip --offload-arch=gfx1100 --offload-arch=gfx1101 \
// RUN:   --no-offload-new-driver -nogpuinc -nogpulib \
// RUN:   %S/Inputs/hip_multiple_inputs/a.cu \
// RUN:   --offload-compress \
// RUN: 2>&1 | FileCheck -check-prefix=CO %s

// CO: clang-offload-bundler{{.*}} "-type=o"
// CO-SAME: -targets={{.*}}hipv4-amdgcn-amd-amdhsa--gfx1100,hipv4-amdgcn-amd-amdhsa--gfx1101
// CO-SAME: "-compress" "-verbose"
