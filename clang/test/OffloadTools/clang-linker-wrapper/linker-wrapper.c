// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -o %t.nvptx.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc
// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -o %t.spirv.bc

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm-bc -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK

// NVPTX-LINK: clang{{.*}} -o {{.*}}.img -dumpdir a.out.nvptx64.sm_70.img. --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.o {{.*}}.o

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --device-compiler=-g \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=NVPTX-LINK-DEBUG

// NVPTX-LINK-DEBUG: clang{{.*}} --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}-g

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LINK

// AMDGPU-LINK: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx908.img. --target=amdgcn-amd-amdhsa -mcpu=gfx908 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030 \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --device-compiler=--save-temps \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-LTO-TEMPS

// AMDGPU-LTO-TEMPS: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}-save-temps

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=SPIRV-LINK

// SPIRV-LINK: clang{{.*}} -o {{.*}}.img -dumpdir a.out.spirv64..img. --target=spirv64-unknown-unknown {{.*}}.o --sycl-link -Xlinker -triple=spirv64-unknown-unknown -Xlinker -arch=

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld --whole-archive %t.a --no-whole-archive \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CPU-LINK

// CPU-LINK: clang{{.*}} -o {{.*}}.img -dumpdir a.out.x86_64..img. --target=x86_64-unknown-linux-gnu -Wl,--no-undefined {{.*}}.o {{.*}}.o -Wl,-Bsymbolic -shared -Wl,--whole-archive {{.*}}.a -Wl,--no-whole-archive

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu -mllvm -openmp-opt-disable \
// RUN:   --linker-path=/usr/bin/ld.lld -a -b -c %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HOST-LINK

// HOST-LINK: ld.lld{{.*}}-a -b -c {{.*}}.o -o a.out
// HOST-LINK-NOT: ld.lld{{.*}}-abc

// RUN: llvm-offload-binary -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-obj.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.a %t-obj.o -o a.out 2>&1 | FileCheck %s --check-prefix=STATIC-LIBRARY

// STATIC-LIBRARY: clang{{.*}} -march=sm_70
// STATIC-LIBRARY-NOT: clang{{.*}} -march=sm_50

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN: --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: clang{{.*}} -o [[IMG_SM70:.+]] -dumpdir a.out.nvptx64.sm_70.img. --target=nvptx64-nvidia-cuda -march=sm_70
// CUDA: clang{{.*}} -o [[IMG_SM52:.+]] -dumpdir a.out.nvptx64.sm_52.img. --target=nvptx64-nvidia-cuda -march=sm_52
// CUDA: fatbinary{{.*}}-64 --create {{.*}}.fatbin --image3=kind=elf,sm=70,file=[[IMG_SM70]] --image3=kind=elf,sm=52,file=[[IMG_SM52]]
// CUDA: usr/bin/ld{{.*}} {{.*}}.openmp.image.{{.*}}.o {{.*}}.cuda.image.{{.*}}.o

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_80 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_75 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_52
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --wrapper-jobs=4 \
// RUN: --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA-PAR
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu --wrapper-jobs=jobserver \
// RUN: --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA-PAR

// CUDA-PAR: fatbinary{{.*}}-64 --create {{.*}}.fatbin

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --compress --compression-level=6 \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

// HIP: clang{{.*}} -o [[IMG_GFX90A:.+]] -dumpdir a.out.amdgcn.gfx90a.img. --target=amdgcn-amd-amdhsa -mcpu=gfx90a
// HIP: clang{{.*}} -o [[IMG_GFX908:.+]] -dumpdir a.out.amdgcn.gfx908.img. --target=amdgcn-amd-amdhsa -mcpu=gfx908
// HIP: clang-offload-bundler{{.*}}-type=o -bundle-align=4096 -compress -compression-level=6 -targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--gfx90a,hip-amdgcn-amd-amdhsa--gfx908 -input={{/dev/null|NUL}} -input=[[IMG_GFX90A]] -input=[[IMG_GFX908]] -output={{.*}}.hipfb

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld --device-linker=foo=bar --device-linker=a \
// RUN:   --device-linker=nvptx64-nvidia-cuda=b --device-compiler=foo\
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=LINKER-ARGS

// LINKER-ARGS: clang{{.*}}--target=amdgcn-amd-amdhsa{{.*}}-Xlinker foo=bar{{.*}}-Xlinker a{{.*}}foo
// LINKER-ARGS: clang{{.*}}--target=nvptx64-nvidia-cuda{{.*}}-Xlinker foo=bar{{.*}}-Xlinker a -Xlinker b{{.*}}foo

// RUN: not clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   -ldummy --linker-path=/usr/bin/ld \
// RUN:   -o a.out 2>&1 | FileCheck %s --check-prefix=MISSING-LIBRARY

// MISSING-LIBRARY: error: unable to find library -ldummy

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --clang-backend \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CLANG-BACKEND

// CLANG-BACKEND: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx908.img. --target=amdgcn-amd-amdhsa -mcpu=gfx908 -Wl,--no-undefined {{.*}}.o

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-windows-msvc -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-windows-msvc --dry-run \
// RUN:   --linker-path=/usr/bin/lld-link %t.o -libpath:./ -out:a.exe 2>&1 | FileCheck %s --check-prefix=COFF

// COFF: "/usr/bin/lld-link" {{.*}}.o -libpath:./ -out:a.exe {{.*}}openmp.image.wrapper{{.*}}

// RUN: llvm-offload-binary -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: llvm-offload-binary -o %t-on.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack+
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-on.o -fembed-offload-object=%t-on.out
// RUN: llvm-offload-binary -o %t-off.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack-
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-off.o -fembed-offload-object=%t-off.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-on.o %t-off.o %t.a -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-TARGET-ID

// AMD-TARGET-ID: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx90a:xnack+.img. --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ -Wl,--no-undefined {{.*}}.o {{.*}}.o
// AMD-TARGET-ID: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx90a:xnack-.img. --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack- -Wl,--no-undefined {{.*}}.o {{.*}}.o

// RUN: llvm-offload-binary -o %t-generic.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-generic.o -fembed-offload-object=%t-generic.out
// RUN: llvm-offload-binary -o %t-on.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack+
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-on.o -fembed-offload-object=%t-on.out
// RUN: llvm-offload-binary -o %t-off.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack-
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-off.o -fembed-offload-object=%t-off.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-on.o %t-off.o %t-generic.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-XNACK-SPLIT

// The three objects are distinct targets (xnack+, xnack-, and feature
// unspecified) and are each linked into their own device image; a
// feature-unspecified object is not merged into a feature-specific image.
// AMD-XNACK-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack+ -Wl,--no-undefined {{.*}}.o
// AMD-XNACK-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a:xnack- -Wl,--no-undefined {{.*}}.o
// AMD-XNACK-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{.*}}.o

// A feature-unspecified (sramecc "any") object and an explicit sramecc+ object
// are distinct targets and are linked into separate device images.
// RUN: llvm-offload-binary -o %t-secc-on.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:sramecc+
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-secc-on.o -fembed-offload-object=%t-secc-on.out
// RUN: llvm-offload-binary -o %t-secc-any.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-secc-any.o -fembed-offload-object=%t-secc-any.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-secc-on.o %t-secc-any.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-SRAMECC-ANY-SPLIT

// AMD-SRAMECC-ANY-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc+ -Wl,--no-undefined {{.*}}.o
// AMD-SRAMECC-ANY-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{.*}}.o

// Objects with explicit conflicting sramecc settings and a feature-unspecified
// object are all distinct targets, each in its own device image.
// RUN: llvm-offload-binary -o %t-generic.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-generic.o -fembed-offload-object=%t-generic.out
// RUN: llvm-offload-binary -o %t-secc-on.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:sramecc+
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-secc-on.o -fembed-offload-object=%t-secc-on.out
// RUN: llvm-offload-binary -o %t-secc-off.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a:sramecc-
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-secc-off.o -fembed-offload-object=%t-secc-off.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-secc-on.o %t-secc-off.o %t-generic.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-SRAMECC-SPLIT

// AMD-SRAMECC-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc+ -Wl,--no-undefined {{.*}}.o
// AMD-SRAMECC-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a:sramecc- -Wl,--no-undefined {{.*}}.o
// AMD-SRAMECC-SPLIT-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{.*}}.o

// Distinct processors, including a specific processor and a member of its
// generic family (gfx900 and gfx9-generic), are separate compilation targets
// and are never merged: each is linked into its own device image. Merging them
// would combine device code from different translation units and produce
// duplicate symbols.
// RUN: llvm-offload-binary -o %t-gfx900.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx900
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-gfx900.o -fembed-offload-object=%t-gfx900.out
// RUN: llvm-offload-binary -o %t-gfx9gen.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx9-generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-gfx9gen.o -fembed-offload-object=%t-gfx9gen.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-gfx900.o %t-gfx9gen.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-DISTINCT-PROC

// AMD-DISTINCT-PROC-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx900 -Wl,--no-undefined {{.*}}.o
// AMD-DISTINCT-PROC-DAG: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx9-generic -Wl,--no-undefined {{.*}}.o

// RUN: llvm-offload-binary -o %t-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: llvm-offload-binary -o %t1.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t1.o -fembed-offload-object=%t1.out
// RUN: llvm-offload-binary -o %t2.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t2.o -fembed-offload-object=%t2.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t1.o %t2.o %t.a -o a.out 2>&1 | FileCheck %s --check-prefix=ARCH-ALL

// ARCH-ALL: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx90a.img. --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{.*}}.o {{.*}}.o
// ARCH-ALL: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx908.img. --target=amdgcn-amd-amdhsa -mcpu=gfx908 -Wl,--no-undefined {{.*}}.o {{.*}}.o

// A static-archive member whose target exactly matches an object group is
// extracted into that group only once, not duplicated (the group is not treated
// as a separate compatible target from the member itself).
// RUN: llvm-offload-binary -o %t-exact-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-exact-libobj.o -fembed-offload-object=%t-exact-lib.out
// RUN: llvm-ar rcs %t-exact.a %t-exact-libobj.o
// RUN: llvm-offload-binary -o %t-exact-obj.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-exact-obj.o -fembed-offload-object=%t-exact-obj.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-exact-obj.o %t-exact.a -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-ARCHIVE-EXACT

// AMD-ARCHIVE-EXACT: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{[^ ]+}}.o {{[^ ]+}}.o{{$}}
// AMD-ARCHIVE-EXACT-NOT: clang{{.*}} -mcpu=gfx90a{{.*}}.o {{.*}}.o {{.*}}.o

// An object names the target as amdgpu9.0a-amd-amdhsa while a static-archive
// member names the same target through the legacy triple plus arch=gfx90a.
// RUN: llvm-offload-binary -o %t-subexact-lib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-subexact-libobj.o -fembed-offload-object=%t-subexact-lib.out
// RUN: llvm-ar rcs %t-subexact.a %t-subexact-libobj.o
// RUN: llvm-offload-binary -o %t-subexact-obj.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9.0a-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-subexact-obj.o -fembed-offload-object=%t-subexact-obj.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-subexact-obj.o %t-subexact.a -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-ARCHIVE-SUBARCH-EXACT

// AMD-ARCHIVE-SUBARCH-EXACT: clang{{.*}} --target=amdgpu9.0a-amd-amdhsa -Wl,--no-undefined {{[^ ]+}}.o {{[^ ]+}}.o{{$}}
// AMD-ARCHIVE-SUBARCH-EXACT-NOT: clang{{.*}} --target={{.*}}-amd-amdhsa -Wl,--no-undefined {{.*}}.o {{.*}}.o {{.*}}.o

// A generic (major-family) archive member provides for a specific member of its
// family: an amdgpu9 (gfx9-generic) archive member is extracted into a specific
// gfx900 object group.
// RUN: llvm-offload-binary -o %t-genlib.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-genlibobj.o -fembed-offload-object=%t-genlib.out
// RUN: llvm-ar rcs %t-gen.a %t-genlibobj.o
// RUN: llvm-offload-binary -o %t-gen-obj.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx900
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-gen-obj.o -fembed-offload-object=%t-gen-obj.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-gen-obj.o %t-gen.a -o a.out 2>&1 | FileCheck %s --check-prefix=AMD-ARCHIVE-GENERIC

// AMD-ARCHIVE-GENERIC: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx900 -Wl,--no-undefined {{[^ ]+}}.o {{[^ ]+}}.o{{$}}
// AMD-ARCHIVE-GENERIC-NOT: clang{{.*}} --target=amdgpu9-amd-amdhsa

// Two images with the same ISA but different triple spellings (the legacy
// "amdgcn" alias and the canonical "amdgpu") must merge into a single device
// link, not produce two separate device images.
// RUN: llvm-offload-binary -o %t-legacy.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-legacy.o -fembed-offload-object=%t-legacy.out
// RUN: llvm-offload-binary -o %t-canon.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-canon.o -fembed-offload-object=%t-canon.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-legacy.o %t-canon.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-TRIPLE-MERGE

// AMDGPU-TRIPLE-MERGE: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined {{.*}}.o {{.*}}.o
// AMDGPU-TRIPLE-MERGE-NOT: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx90a -Wl,--no-undefined

// Two images whose triples encode the ISA in the subarch (amdgpu9.0a-amd-amdhsa)
// and carry no separate arch must merge into a single device link.
// RUN: llvm-offload-binary -o %t-sub1.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9.0a-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-sub1.o -fembed-offload-object=%t-sub1.out
// RUN: llvm-offload-binary -o %t-sub2.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9.0a-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-sub2.o -fembed-offload-object=%t-sub2.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-sub1.o %t-sub2.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-SUBARCH-MERGE

// AMDGPU-SUBARCH-MERGE: clang{{.*}} --target=amdgpu9.0a-amd-amdhsa -Wl,--no-undefined {{.*}}.o {{.*}}.o
// AMDGPU-SUBARCH-MERGE-NOT: clang{{.*}} --target=amdgpu9.0a-amd-amdhsa -Wl,--no-undefined

// A major-family subarch triple (amdgpu9-amd-amdhsa) and a specific member of
// that family (amdgpu9.00-amd-amdhsa) are distinct compilation targets: each
// object carries its own device code, so they must be linked into separate
// device images rather than merged (merging would duplicate symbols).
// RUN: llvm-offload-binary -o %t-specific.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9.00-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-specific.o -fembed-offload-object=%t-specific.out
// RUN: llvm-offload-binary -o %t-major.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgpu9-amd-amdhsa
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-major.o -fembed-offload-object=%t-major.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-specific.o %t-major.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-MAJOR-SUBARCH-SPLIT

// AMDGPU-MAJOR-SUBARCH-SPLIT-DAG: clang{{.*}} --target=amdgpu9.00-amd-amdhsa -Wl,--no-undefined {{.*}}.o
// AMDGPU-MAJOR-SUBARCH-SPLIT-DAG: clang{{.*}} --target=amdgpu9-amd-amdhsa -Wl,--no-undefined {{.*}}.o

// Two distinct GPUs in the same major family are not interchangeable and must
// not merge.
// RUN: llvm-offload-binary -o %t-gfx900.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx900
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-gfx900.o -fembed-offload-object=%t-gfx900.out
// RUN: llvm-offload-binary -o %t-gfx906.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx906
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t-gfx906.o -fembed-offload-object=%t-gfx906.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t-gfx900.o %t-gfx906.o -o a.out 2>&1 | FileCheck %s --check-prefix=AMDGPU-SAME-MAJOR-NO-MERGE

// AMDGPU-SAME-MAJOR-NO-MERGE-DAG: clang{{.*}} -mcpu=gfx900 -Wl,--no-undefined {{.*}}.o
// AMDGPU-SAME-MAJOR-NO-MERGE-DAG: clang{{.*}} -mcpu=gfx906 -Wl,--no-undefined {{.*}}.o

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=x86_64-unknown-linux-gnu
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK

// RELOCATABLE-LINK: clang{{.*}} -o {{.*}}.img -dumpdir a.out.x86_64..img. --target=x86_64-unknown-linux-gnu
// RELOCATABLE-LINK: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK-HIP

// RELOCATABLE-LINK-HIP: clang{{.*}} -o {{.*}}.img -dumpdir a.out.amdgcn.gfx90a.img. --target=amdgcn-amd-amdhsa
// RELOCATABLE-LINK-HIP: clang-offload-bundler{{.*}} -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--gfx90a -input={{/dev/null|NUL}} -input={{.*}} -output={{.*}}
// RELOCATABLE-LINK-HIP: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK-HIP: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading
// RELOCATABLE-LINK-HIP: --rename-section llvm_offload_entries

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_89 \
// RUN:   --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_89
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld -r %t.o \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=RELOCATABLE-LINK-CUDA

// RELOCATABLE-LINK-CUDA: clang{{.*}} -o {{.*}}.img -dumpdir a.out.nvptx64.sm_89.img. --target=nvptx64-nvidia-cuda
// RELOCATABLE-LINK-CUDA: fatbinary{{.*}} -64 --create {{.*}}.fatbin --image3=kind=elf,sm=89,file={{.*}}.img
// RELOCATABLE-LINK-CUDA: /usr/bin/ld.lld{{.*}}-r
// RELOCATABLE-LINK-CUDA: llvm-objcopy{{.*}}a.out --remove-section .llvm.offloading

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld --override=image=openmp=%t.o %t.o -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=OVERRIDE
// OVERRIDE-NOT: clang
// OVERRIDE: /usr/bin/ld

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --offload-opt=-pass-remarks=foo,bar --linker-path=/usr/bin/ld \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OFFLOAD-OPT
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   -mllvm -pass-remarks=foo,bar --linker-path=/usr/bin/ld \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=MLLVM

//            MLLVM: clang{{.*}}-Xlinker --plugin-opt=-pass-remarks=foo,bar
//      OFFLOAD-OPT: clang{{.*}}-Xlinker --plugin-opt=-pass-remarks=foo,bar
//       MLLVM-SAME: -Xlinker -mllvm=-pass-remarks=foo,bar
//  OFFLOAD-OPT-NOT: -Xlinker -mllvm=-pass-remarks=foo,bar
// OFFLOAD-OPT-SAME: {{$}}

// Error handling when --linker-path is not provided for clang-linker-wrapper
// RUN: not clang-linker-wrapper 2>&1 | FileCheck --check-prefix=LINKER-PATH-NOT-PROVIDED %s
// LINKER-PATH-NOT-PROVIDED: linker path missing, must pass 'linker-path'
