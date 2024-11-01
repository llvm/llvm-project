// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.nvptx.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK

// NVPTX-LINK: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --device-debug -O0 \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK-DEBUG

// NVPTX-LINK-DEBUG: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 -O2 -Wl,--no-undefined -g {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LINK

// AMDGPU-LINK: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030 \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --save-temps -O2 \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LTO-TEMPS

// AMDGPU-LTO-TEMPS: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -O2 -Wl,--no-undefined -save-temps {{.*}}.s

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CPU-LINK

// CPU-LINK: clang{{.*}} -o {{.*}}.img --target=x86_64-unknown-linux-gnu -march=native -O2 -Wl,--no-undefined -Bsymbolic -shared {{.*}}.o {{.*}}.o

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu -mllvm -openmp-opt-disable \
// RUN:   --linker-path=/usr/bin/ld.lld -- -a -b -c %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HOST-LINK

// HOST-LINK: ld.lld{{.*}}-a -b -c {{.*}}.o -o a.out
// HOST-LINK-NOT: ld.lld{{.*}}-abc

// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-obj.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld -- %t.a %t-obj.o -o a.out 2>&1 | FileCheck %s --check-prefix=STATIC-LIBRARY

// STATIC-LIBRARY: clang{{.*}} -march=sm_70
// STATIC-LIBRARY-NOT: clang{{.*}} -march=sm_50

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN: --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: clang{{.*}} -o [[IMG_SM52:.+]] --target=nvptx64-nvidia-cuda -march=sm_52
// CUDA: clang{{.*}} -o [[IMG_SM70:.+]] --target=nvptx64-nvidia-cuda -march=sm_70
// CUDA: fatbinary{{.*}}-64 --create {{.*}}.fatbin --image=profile=sm_70,file=[[IMG_SM70]] --image=profile=sm_52,file=[[IMG_SM52]] 

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_80 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_75 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --wrapper-jobs=4 \
// RUN: --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA-PAR

// CUDA-PAR: fatbinary{{.*}}-64 --create {{.*}}.fatbin

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

// HIP: clang{{.*}} -o [[IMG_GFX908:.+]] --target=amdgcn-amd-amdhsa -mcpu=gfx908
// HIP: clang{{.*}} -o [[IMG_GFX90A:.+]] --target=amdgcn-amd-amdhsa -mcpu=gfx90a
// HIP: clang-offload-bundler{{.*}}-type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a,hipv4-amdgcn-amd-amdhsa--gfx908 -input=/dev/null -input=[[IMG_GFX90A]] -input=[[IMG_GFX908]] -output={{.*}}.hipfb

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --device-linker=a --device-linker=nvptx64-nvidia-cuda=b -- \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LINKER-ARGS

// LINKER-ARGS: clang{{.*}}--target=amdgcn-amd-amdhsa{{.*}}-Wl,a
// LINKER-ARGS: clang{{.*}}--target=nvptx64-nvidia-cuda{{.*}}-Wl,a -Wl,b

// RUN: not clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu -ldummy \
// RUN:   --linker-path=/usr/bin/ld --device-linker=a --device-linker=nvptx64-nvidia-cuda=b -- \
// RUN:   -o a.out 2>&1 | FileCheck %s --check-prefix=MISSING-LIBRARY

// MISSING-LIBRARY: error: unable to find library -ldummy
