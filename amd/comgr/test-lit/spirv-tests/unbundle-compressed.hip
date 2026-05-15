// REQUIRES: comgr-has-spirv-translator

// COM: Build a compressed bitcode bundle for amdgcnspirv.
// RUN: %clang -c -x hip --offload-arch=amdgcnspirv \
// RUN:    -nogpulib -nogpuinc -emit-llvm \
// RUN:    --gpu-bundle-output --offload-device-only \
// RUN:    --offload-compress \
// RUN:    %s -o %t.compressed-bundle.bc

// COM: Sanity-check that the bundle is actually compressed (CCOB magic) and
// COM: that the expected entry ID is present.
// RUN: od -An -c -N 4 %t.compressed-bundle.bc | %FileCheck --check-prefix=MAGIC %s
// RUN: clang-offload-bundler --type=bc --input=%t.compressed-bundle.bc --list \
// RUN:   | %FileCheck --check-prefix=ENTRY %s

// MAGIC: C C O B
// ENTRY: hip-spirv64-amd-amdhsa--amdgcnspirv

// COM: Extract the amdgcnspirv entry through Comgr. Output is a SPIR-V module
// COM: (not LLVM bitcode), so verify the SPIR-V magic word 0x07230203.
// RUN: unbundle %t.compressed-bundle.bc \
// RUN:   hip-spirv64-amd-amdhsa--amdgcnspirv %t.amdgcnspirv.spv
// RUN: od -An -tx4 -N 4 %t.amdgcnspirv.spv | %FileCheck --check-prefix=SPV %s

// SPV: 07230203

// COM: Translate the unbundled SPIR-V back to LLVM IR via Comgr's translator
// COM: and verify it matches the original device source.
// RUN: spirv-translator %t.amdgcnspirv.spv -o %t.translated.bc
// RUN: %llvm-dis %t.translated.bc -o - | %FileCheck --check-prefix=BC %s

// BC: target triple = "amdgcn-amd-amdhsa"
// BC: define {{.*}} @_Z9add_valuePfS_S_

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
