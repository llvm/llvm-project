// REQUIRES: x86-registered-target
// REQUIRES: zlib || zstd

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o

// RUN: llvm-offload-binary -o %t.out --compress \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgpu9.0a-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgpu9.08-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=CHECK,HIP

// RUN: llvm-offload-binary -o %t-lib.out --compress \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9.0a-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-lib.o \
// RUN:   -fembed-offload-object=%t-lib.out
// RUN: rm -f %t.a && llvm-ar rcs %t.a %t-lib.o
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --whole-archive %t.a --no-whole-archive \
// RUN:   -o a.out 2>&1 | FileCheck %s

// CHECK: clang{{.*}} --target=amdgpu9.0a-amd-amdhsa -mcpu=gfx90a
// HIP:   clang{{.*}} --target=amdgpu9.08-amd-amdhsa -mcpu=gfx908
