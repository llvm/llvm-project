// Create compressed bitcode bundle (add --offload-compress flag)
// RUN: clang -c -x hip --offload-arch=gfx900 --offload-arch=gfx1030 \
// RUN:    -nogpulib -nogpuinc \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    -emit-llvm \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.bc
//
// Clean the cache
// RUN: rm -fr %t.cache
//
// With the cache enabled, test that we write one file to the cache
// RUN: export AMD_COMGR_CACHE=1
// RUN: export AMD_COMGR_CACHE_DIR=%t.cache
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.cache_1.bc
// RUN: llvm-dis %t.cache_1.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
//
// If there is a re-run, the cache contents remain the same
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx900 \
// RUN:    %t.cache_2.bc
// RUN: llvm-dis %t.cache_2.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 2 -eq $COUNT ]
//
// A run with different input options results in new contents in the cache
// RUN: unbundle %t.compressed-bundle.bc hip-amdgcn-amd-amdhsa-unknown-gfx1030 \
// RUN:    %t.cache_3.bc
// RUN: llvm-dis %t.cache_3.bc -o - | FileCheck --check-prefixes=BOTH,GFX10 %s
// RUN: COUNT=$(ls "%t.cache" | wc -l)
// RUN: [ 3 -eq $COUNT ]

// BOTH: target triple = "amdgcn-amd-amdhsa"
// GFX9: "target-cpu"="gfx900"
// GFX10: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
