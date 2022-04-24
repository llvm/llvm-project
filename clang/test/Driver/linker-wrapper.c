// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,nvptx64-nvida-cuda,sm_70 \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,nvptx64-nvida-cuda,sm_70
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX_LINK

// NVPTX_LINK: nvlink{{.*}}-m64 -o {{.*}}.out -arch sm_70 {{.*}}.o {{.*}}.o

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,amdgcn-amd-amdhsa,gfx908 \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,amdgcn-amd-amdhsa,gfx908
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU_LINK

// AMDGPU_LINK: lld{{.*}}-flavor gnu --no-undefined -shared -o {{.*}}.out {{.*}}.o {{.*}}.o

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,x86_64-unknown-linux-gnu, \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,x86_64-unknown-linux-gnu,
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld.lld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CPU_LINK

// CPU_LINK: ld.lld{{.*}}-m elf_x86_64 -shared -Bsymbolic -o {{.*}}.out {{.*}}.o {{.*}}.o

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple x86_64-unknown-linux-gnu -linker-path \
// RUN:  /usr/bin/ld.lld -- -a -b -c %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HOST_LINK

// HOST_LINK: ld.lld{{.*}}-a -b -c {{.*}}.o -o a.out

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-bc.bc,openmp,nvptx64-nvida-cuda,sm_70 \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-bc.bc,openmp,nvptx64-nvida-cuda,sm_70
// RUN: clang-linker-wrapper --host-triple x86_64-unknown-linux-gnu --dry-run -linker-path \
// RUN:   /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LTO

// LTO: ptxas{{.*}}-m64 -o {{.*}}.cubin -O2 --gpu-name sm_70 {{.*}}.s
// LTO-NOT: nvlink
