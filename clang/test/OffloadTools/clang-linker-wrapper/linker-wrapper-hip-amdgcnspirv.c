// RUN: %clang -cc1 %s -triple "spirv64-amd-amdhsa" -emit-llvm-bc -o %t.bc
// RUN: llvm-offload-binary -o %t.out "--image=file=%t.bc,triple=spirv64-amd-amdhsa,arch=amdgcnspirv,kind=hip"
// RUN: clang-linker-wrapper \
// RUN:     "--should-extract=amdgcnspirv" \
// RUN:     "--host-triple=spirv64-amd-amdhsa" \
// RUN:     "--linker-path=clang-offload-bundler" \
// RUN:     "--emit-fatbin-only" \
// RUN:     "-o" "%t.hipfb" \
// RUN:     "%t.out" \
// RUN:     --dry-run \
// RUN: 2>&1 | FileCheck %s

// clang-linker-wrapper was previously calling clang-offload-bundler with -targets=...,hip-amdgpu-amd-amdhsa--amdgcnspirv
// This caused the runtime not to recognise the triple for the AMD SPIR-V code.

// CHECK: {{".*clang-offload-bundler.*"}} {{.*}} -targets={{.*}},hip-spirv64-amd-amdhsa--amdgcnspirv

// The host bundle entry must be a four component triple.
// RUN: llvm-offload-binary -o %t.host.out \
// RUN:   --image=file=%s,kind=hip,triple=amdgpu9.0a-amd-amdhsa,arch=gfx90a
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-apple-macosx15.0.0 \
// RUN:   --emit-fatbin-only %t.host.out -o %t.host.hipfb 2>&1 | FileCheck %s --check-prefix=HOST-TRIPLE
// HOST-TRIPLE: clang-offload-bundler{{.*}} -targets=host-x86_64-apple-macosx15.0.0-unknown,hip-amdgcn-amd-amdhsa--gfx90a
