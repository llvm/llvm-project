// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: system-linux

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out

// RUN: clang-linker-wrapper --lto-opt-pipeline=default \
// RUN:   --dry-run --wrapper-verbose --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=LTO-OPT-PL-00
// LTO-OPT-PL-00: "{{.*}}clang" {{.*}} -Xlinker --lto-newpm-passes=default<O2>

// RUN: clang-linker-wrapper --lto-opt-pipeline=default --opt-level=O3 \
// RUN:   --dry-run --wrapper-verbose --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=LTO-OPT-PL-01
// LTO-OPT-PL-01: "{{.*}}clang" {{.*}} -Xlinker --lto-newpm-passes=default<O3>

// RUN: clang-linker-wrapper --lto-opt-pipeline=lto \
// RUN:   --dry-run --wrapper-verbose --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=LTO-OPT-PL-02
// LTO-OPT-PL-02: "{{.*}}clang" {{.*}} -Xlinker --lto-newpm-passes=lto<O2>

// RUN: clang-linker-wrapper --lto-opt-pipeline=lto --opt-level=O0 \
// RUN:   --dry-run --wrapper-verbose --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=LTO-OPT-PL-03
// LTO-OPT-PL-03: "{{.*}}clang" {{.*}} -Xlinker --lto-newpm-passes=lto<O0>

// RUN: clang-linker-wrapper \
// RUN:   --dry-run --wrapper-verbose --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out \
// RUN:   2>&1 | FileCheck %s --check-prefix=LTO-OPT-PL-04
// LTO-OPT-PL-04-NOT: --lto-newpm-passes
