// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o

#if defined(RESOLVES)
int __attribute__((visibility("hidden"))) sym;
#elif defined(GLOBAL)
int __attribute__((visibility("protected"))) global;
#elif defined(WEAK)
int __attribute__((visibility("hidden"))) weak;
#elif defined(HIDDEN)
int __attribute__((visibility("hidden"))) hidden;
#elif defined(UNDEFINED)
extern int sym;
int baz() { return sym; }
#else
extern int sym;

extern int __attribute__((weak)) weak;

int foo() { return sym; }
int bar() { return weak; }
#endif

//
// Check that we extract a static library defining an undefined symbol.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DRESOLVES -o %t.nvptx.resolves.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DRESOLVES -o %t.amdgpu.resolves.bc
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DUNDEFINED -o %t.nvptx.undefined.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DUNDEFINED -o %t.amdgpu.undefined.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.undefined.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.undefined.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030 \
// RUN:   --image=file=%t.nvptx.resolves.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.resolves.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a -o a.out 2>&1 \
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.a %t.o -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-RESOLVES

// LIBRARY-RESOLVES: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}.o {{.*}}.o
// LIBRARY-RESOLVES: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.s {{.*}}.o

//
// Check that we extract a static library that defines a global visibile to the
// host.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DGLOBAL -o %t.nvptx.global.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DGLOBAL -o %t.amdgpu.global.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.global.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.global.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a -o a.out 2>&1 \
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.a %t.o -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-GLOBAL

// LIBRARY-GLOBAL: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}.o {{.*}}.o
// LIBRARY-GLOBAL: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.s {{.*}}.o

//
// Check that we do not extract a global symbol if the source file was not
// created by an offloading language that expects there to be a host version of
// the symbol.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DGLOBAL -o %t.nvptx.global.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DGLOBAL -o %t.amdgpu.global.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.global.bc,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.global.bc,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-GLOBAL-NONE

// LIBRARY-GLOBAL-NONE-NOT: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}.o {{.*}}.o
// LIBRARY-GLOBAL-NONE-NOT: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.s {{.*}}.o

//
// Check that we do not extract an external weak symbol.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DWEAK -o %t.nvptx.weak.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DWEAK -o %t.amdgpu.weak.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.weak.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.weak.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-WEAK

// LIBRARY-WEAK: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030
// LIBRARY-WEAK-NOT: {{.*}}.o {{.*}}.o
// LIBRARY-WEAK: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70

//
// Check that we do not extract an unneeded hidden symbol.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DHIDDEN -o %t.nvptx.hidden.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DHIDDEN -o %t.amdgpu.hidden.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.hidden.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.hidden.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-HIDDEN

// LIBRARY-HIDDEN: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030
// LIBRARY-HIDDEN-NOT: {{.*}}.o {{.*}}.o
// LIBRARY-HIDDEN: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70

//
// Check that we do not extract a static library that defines a global visibile
// to the host that is already defined.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DGLOBAL -o %t.nvptx.global.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DGLOBAL -o %t.amdgpu.global.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.global.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.amdgpu.global.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o %t.a %t.a -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-GLOBAL-DEFINED

// LIBRARY-GLOBAL-DEFINED: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}.o {{.*}}.o
// LIBRARY-GLOBAL-DEFINED-NOT: {{.*}}gfx1030{{.*}}.o
// LIBRARY-GLOBAL-DEFINED: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.s {{.*}}.o

//
// Check that we can use --[no-]whole-archive to control extraction.
//
// RUN: %clang -cc1 %s -triple nvptx64-nvidia-cuda -emit-llvm-bc -DGLOBAL -o %t.nvptx.global.bc
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -DGLOBAL -o %t.amdgpu.global.bc
// RUN: clang-offload-packager -o %t-lib.out \
// RUN:   --image=file=%t.nvptx.global.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.nvptx.global.bc,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_52 \
// RUN:   --image=file=%t.amdgpu.global.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030 \
// RUN:   --image=file=%t.amdgpu.global.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t-lib.out
// RUN: llvm-ar rcs %t.a %t.o
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx1030
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.o --whole-archive %t.a -o a.out 2>&1 \
// RUN: | FileCheck %s --check-prefix=LIBRARY-WHOLE-ARCHIVE

// LIBRARY-WHOLE-ARCHIVE: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx1030 {{.*}}.o {{.*}}.o
// LIBRARY-WHOLE-ARCHIVE: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_52 {{.*}}.s
// LIBRARY-WHOLE-ARCHIVE: clang{{.*}} -o {{.*}}.img --target=amdgcn-amd-amdhsa -mcpu=gfx90a {{.*}}.o
// LIBRARY-WHOLE-ARCHIVE: clang{{.*}} -o {{.*}}.img --target=nvptx64-nvidia-cuda -march=sm_70 {{.*}}.s {{.*}}.o
