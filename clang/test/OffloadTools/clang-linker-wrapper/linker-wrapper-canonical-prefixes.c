// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: rm -rf %t.test_dir && mkdir -p %t.test_dir
// RUN: touch %t.test_dir/clang
// RUN: chmod +x %t.test_dir/clang
// RUN: ln -s clang-linker-wrapper %t.test_dir/clang-linker-wrapper

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: %t.test_dir/clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out --no-canonical-prefixes 2>&1 | FileCheck %s

// Check that we resolve clang to the symlink rather than the bin/ directory
// and that the sub-clang invocation was passed -no-canonical-prefixes.
// CHECK: test_dir/clang"
