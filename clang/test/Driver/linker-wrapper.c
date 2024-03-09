// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// REQUIRES: system-linux

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.nvptx.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK

// NVPTX-LINK: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 -O2 {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --device-debug -O0 \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK-DEBUG

// NVPTX-LINK-DEBUG: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 -O2 {{.*}}.o {{.*}}.o -g 

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LINK

// AMDGPU-LINK: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030 \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --save-temps -O2 \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LTO-TEMPS

// AMDGPU-LTO-TEMPS: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 -O2 -Wl,--no-undefined {{.*}}.s -save-temps

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld --whole-archive %t.a --no-whole-archive \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CPU-LINK

// CPU-LINK: clang{{.*}} -o {{.*}}.img --target=x86_64-unknown-linux-gnu -march=native -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o -Wl,-Bsymbolic -shared -Wl,--whole-archive {{.*}}.a -Wl,--no-whole-archive

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu -mllvm -openmp-opt-disable \
// RUN:   --linker-path=/usr/bin/ld.lld -a -b -c %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HOST-LINK

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
// RUN:   --linker-path=/usr/bin/ld %t.a %t-obj.o -o a.out 2>&1 | FileCheck %s --check-prefix=STATIC-LIBRARY

// STATIC-LIBRARY: clang{{.*}} -march=sm_70
// STATIC-LIBRARY-NOT: clang{{.*}} -march=sm_50

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN: --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: clang{{.*}} -o [[IMG_SM52:.+]] --target=nvptx64-nvidia-cuda -march=sm_52
// CUDA: clang{{.*}} -o [[IMG_SM70:.+]] --target=nvptx64-nvidia-cuda -march=sm_70
// CUDA: fatbinary{{.*}}-64 --create {{.*}}.fatbin --image=profile=sm_70,file=[[IMG_SM70]] --image=profile=sm_52,file=[[IMG_SM52]] 
// CUDA: usr/bin/ld{{.*}} {{.*}}.openmp.image.{{.*}}.o {{.*}}.cuda.image.{{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_80 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_75 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --wrapper-jobs=4 \
// RUN: --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA-PAR

// CUDA-PAR: fatbinary{{.*}}-64 --create {{.*}}.fatbin

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --compress --compression-level=6 \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

// HIP: clang{{.*}} -o [[IMG_GFX908:.+]] --target=amdgcn-amd-amdhsa -mcpu=gfx908
// HIP: clang{{.*}} -o [[IMG_GFX90A:.+]] --target=amdgcn-amd-amdhsa -mcpu=gfx90a
// HIP: clang-offload-bundler{{.*}}-type=o -bundle-align=4096 -compress -compression-level=6 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a,hipv4-amdgcn-amd-amdhsa--gfx908 -input=/dev/null -input=[[IMG_GFX90A]] -input=[[IMG_GFX908]] -output={{.*}}.hipfb

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --device-linker=a --device-linker=nvptx64-nvidia-cuda=b \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LINKER-ARGS

// LINKER-ARGS: clang{{.*}}--target=amdgcn-amd-amdhsa{{.*}}a
// LINKER-ARGS: clang{{.*}}--target=nvptx64-nvidia-cuda{{.*}}a b

// RUN: not clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu -ldummy \
// RUN:   --linker-path=/usr/bin/ld --device-linker=a --device-linker=nvptx64-nvidia-cuda=b \
// RUN:   -o a.out 2>&1 | FileCheck %s --check-prefix=MISSING-LIBRARY

// MISSING-LIBRARY: error: unable to find library -ldummy

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --clang-backend \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CLANG-BACKEND

// CLANG-BACKEND: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -Wl,--no-undefined {{.*}}.bc

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-windows-msvc -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o -libpath:./ -out:a.exe 2>&1 | FileCheck %s --check-prefix=COFF

// COFF: "/usr/bin/lld-link" {{.*}}.o -libpath:./ -out:a.exe {{.*}}openmp.image.wrapper{{.*}}

// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t-on.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack+
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-on.o -fembed-offload-object=%t-on.out
// RUN: clang-offload-packager -o %t-off.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack-
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-off.o -fembed-offload-object=%t-off.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-on.o %t-off.o %t.a -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-TARGET-ID

// AMD-TARGET-ID: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o
// AMD-TARGET-ID: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack- -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t1.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t1.o -fembed-offload-object=%t1.out
// RUN: clang-offload-packager -o %t2.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t2.o -fembed-offload-object=%t2.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t1.o %t2.o %t.a -o a.out 2>&1 | FileCheck %s --check-prefix=ARCH-ALL

// ARCH-ALL: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx908 -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o
// ARCH-ALL: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx90a -O2 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK

// RELOCATABLE-LINK: clang{{.*}} -o {{.*}}.img --target=x86_64-unknown-linux-gnu
// RELOCATABLE-LINK: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK-HIP

// RELOCATABLE-LINK-HIP: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa
// RELOCATABLE-LINK-HIP: clang-offload-bundler{{.*}} -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input={{.*}} -output={{.*}}
// RELOCATABLE-LINK-HIP: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK-HIP: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading

// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_89 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_89
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK-CUDA

// RELOCATABLE-LINK-CUDA: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda
// RELOCATABLE-LINK-CUDA: fatbinary{{.*}} -64 --create {{.*}}.fatbin --image=profile=sm_89,file={{.*}}.img
// RELOCATABLE-LINK-CUDA: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK-CUDA: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading
