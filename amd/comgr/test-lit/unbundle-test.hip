// Create bitcode bundle
// RUN: clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx1030 \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    %s -o %t.bundle.bc
//
// Create compressed bitcode bundle (add --offload-compress flag)
// RUN: clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx1030 \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.bc
//
// Extract using Comgr
// RUN: unbundle %t.bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 %t.gfx900.bc
// RUN: llvm-dis %t.gfx900.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx1030 %t.compressed.gfx1030.bc
// RUN: llvm-dis %t.compressed.gfx1030.bc -o - | FileCheck --check-prefixes=BOTH,GFX10 %s
//
// BOTH: target triple = "amdgcn-amd-amdhsa"
// GFX9: "target-cpu"="gfx900"
// GFX10: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
