// REQUIRES: x86-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o

//
// For OpenMP everything goes through the LLVM offloading binary type.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

// OPENMP: llvm-offload-binary{{.*}} {{.*}}.o --image=kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70,file={{.*}}.o --image=kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a,file={{.*}}.o
// OPENMP: clang{{.*}} --target=nvptx64-nvidia-cuda -march=sm_70
// OPENMP: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a
// OPENMP: llvm-offload-binary{{.*}} -o {{.*}}.offload --image=file={{.*}}.img,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// OPENMP: llvm-offload-binary{{.*}} -o {{.*}}.offload --image=file={{.*}}.img,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// OPENMP: llvm-offload-wrapper{{.*}} --kind=openmp --triple=x86_64-unknown-linux-gnu -o [[BC:.*]].bc {{.*}}.offload {{.*}}.offload
// OPENMP: clang{{.*}} --no-default-config --target=x86_64-unknown-linux-gnu -c -fPIC -o {{.*}}.openmp.image.wrapper{{.*}}.o [[BC]].bc

//
// The '--relocatable' flag is forwarded to the wrapper tool for OpenMP.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --linker-path=/usr/bin/ld -r %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE

// RELOCATABLE: llvm-offload-wrapper{{.*}} --kind=openmp --triple=x86_64-unknown-linux-gnu -o {{.*}}.bc --relocatable {{.*}}.offload

//
// For CUDA the device images are combined with 'fatbinary'.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: llvm-offload-binary{{.*}} {{.*}}.o --image=kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70,file={{.*}}.o --image=kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52,file={{.*}}.o
// CUDA: clang{{.*}} --target=nvptx64-nvidia-cuda -march=sm_70
// CUDA: clang{{.*}} --target=nvptx64-nvidia-cuda -march=sm_52
// CUDA: fatbinary{{.*}}--create [[FB:.*]].fatbin {{.*}}--image3=kind=elf,sm=70{{.*}}--image3=kind=elf,sm=52
// CUDA: llvm-offload-wrapper{{.*}} --kind=cuda --triple=x86_64-unknown-linux-gnu -o [[BC:.*]].bc [[FB]].fatbin
// CUDA: clang{{.*}} --no-default-config --target=x86_64-unknown-linux-gnu -c -fPIC -o {{.*}}.cuda.image.wrapper{{.*}}.o [[BC]].bc

//
// For HIP the device images are combined with 'clang-offload-bundler'.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

// HIP: llvm-offload-binary{{.*}} {{.*}}.o --image=kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a,file={{.*}}.o --image=kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908,file={{.*}}.o
// HIP: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a
// HIP: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx908
// HIP: clang-offload-bundler{{.*}}-targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--gfx90a,hip-amdgcn-amd-amdhsa--gfx908{{.*}}-output=[[FB:.*]].hipfb
// HIP: llvm-offload-wrapper{{.*}} --kind=hip --triple=x86_64-unknown-linux-gnu -o [[BC:.*]].bc [[FB]].hipfb
// HIP: clang{{.*}} --no-default-config --target=x86_64-unknown-linux-gnu -c -fPIC -o {{.*}}.hip.image.wrapper{{.*}}.o [[BC]].bc

//
// For SYCL the device image is linked with 'clang --sycl-link' and wrapped
// directly with 'llvm-offload-wrapper --kind=sycl'.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=SYCL

// SYCL: llvm-offload-binary{{.*}} {{.*}}.o --image=kind=sycl,triple=spirv64-unknown-unknown,arch=generic,file={{.*}}.o
// SYCL: clang{{.*}} --target=spirv64-unknown-unknown {{.*}} --sycl-link {{.*}}-triple=spirv64-unknown-unknown{{.*}}-arch=
// SYCL: llvm-offload-wrapper{{.*}} --kind=sycl --triple=x86_64-unknown-linux-gnu -o [[BC:.*]].bc {{.*}}.img
// SYCL: clang{{.*}} --no-default-config --target=x86_64-unknown-linux-gnu -c -fPIC -o {{.*}}.sycl.image.wrapper{{.*}}.o [[BC]].bc

//
// Images pulled from a static archive are extracted from the archive path and
// singled out by their member name so the replayed command is reproducible.
//
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: rm -f %t.a && llvm-ar rcs %t.a %t.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --wrapper-verbose --save-temps --dry-run \
// RUN:   --should-extract=sm_70 --linker-path=/usr/bin/ld %t.a -o a.out 2>&1 | FileCheck %s --check-prefix=ARCHIVE

// ARCHIVE: llvm-offload-binary{{.*}} {{.*}}.a --image=kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70,member={{.*}}.o,file={{.*}}.o
