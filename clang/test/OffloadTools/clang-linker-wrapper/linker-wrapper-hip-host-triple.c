// The host bundle entry must be a four component triple.

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%s,kind=hip,triple=amdgpu9.0a-amd-amdhsa,arch=gfx90a
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-apple-macosx15.0.0 \
// RUN:   --emit-fatbin-only %t.out -o %t.hipfb 2>&1 | FileCheck %s

// CHECK: clang-offload-bundler{{.*}} -targets=host-x86_64-apple-macosx15.0.0-unknown,hip-amdgcn-amd-amdhsa--gfx90a
