// REQUIRES: x86-registered-target, amdgpu-registered-target, lld, system-linux

// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu \
// RUN:   --offload-arch=gfx906 -c -nostdinc -nogpuinc -nohipwrapperinc \
// RUN:   -nogpulib -fgpu-rdc -I%S/Inputs %s -o %t.1.o

// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -DLIB \
// RUN:   --offload-arch=gfx906 -c -nostdinc -nogpuinc -nohipwrapperinc \
// RUN:   -nogpulib -fgpu-rdc -I%S/Inputs %s -o %t.2.o

// RUN: %clang -x hip --target=x86_64-unknown-linux-gnu -DMAIN \
// RUN:   --offload-arch=gfx906 -c -nostdinc -nogpuinc -nohipwrapperinc \
// RUN:   -nogpulib -fgpu-rdc -I%S/Inputs %s -o %t.main.o

// RUN: llvm-nm  %t.1.o | FileCheck -check-prefix=OBJ1 %s
// OBJ1:  B __hip_cuid_[[ID:[0-9a-f]+]]
// OBJ1:  U __hip_fatbin_[[ID]]
// OBJ1:  U __hip_gpubin_handle_[[ID]]

// RUN: llvm-nm  %t.2.o | FileCheck -check-prefix=OBJ2 %s
// OBJ2:  B __hip_cuid_[[ID:[0-9a-f]+]]
// OBJ2:  U __hip_fatbin_[[ID]]
// OBJ2:  U __hip_gpubin_handle_[[ID]]

// Link %t.1.o and %t.2.o by -r and then link with %t.main.o

// RUN: %clang -v --target=x86_64-unknown-linux-gnu \
// RUN:   --hip-link -fgpu-rdc --offload-arch=gfx906 \
// RUN:   -r -fuse-ld=lld -nostdlib %t.1.o %t.2.o -o %t.lib.o \
// RUN:   2>&1 | FileCheck -check-prefix=LD-R %s
// LD-R: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID1:[0-9a-f]+]]
// LD-R: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID2:[0-9a-f]+]]
// LD-R: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID1]]
// LD-R: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID2]]
// LD-R: "{{.*}}/clang-offload-bundler" {{.*}}-unbundle
// LD-R: "{{.*}}/lld" -flavor gnu -m elf64_amdgpu
// LD-R: "{{.*}}/clang-offload-bundler"
// LD-R: "{{.*}}/llvm-mc" -triple x86_64-unknown-linux-gnu
// LD-R: "{{.*}}/ld.lld" {{.*}} -r

// RUN: llvm-nm  %t.lib.o | FileCheck -check-prefix=OBJ %s
// OBJ:  B __hip_cuid_[[ID1:[0-9a-f]+]]
// OBJ:  B __hip_cuid_[[ID2:[0-9a-f]+]]
// OBJ:  R __hip_fatbin_[[ID1]]
// OBJ:  R __hip_fatbin_[[ID2]]
// OBJ:  D __hip_gpubin_handle_[[ID1]]
// OBJ:  D __hip_gpubin_handle_[[ID2]]

// RUN: %clang -v --target=x86_64-unknown-linux-gnu \
// RUN:   --hip-link -no-hip-rt -fgpu-rdc --offload-arch=gfx906 \
// RUN:   -fuse-ld=lld -nostdlib -r %t.main.o %t.lib.o -o %t.final.o \
// RUN:   2>&1 | FileCheck -check-prefix=LINK-O %s
// LINK-O-NOT: Found undefined HIP {{.*}}symbol

// Generate a static lib with %t.1.o and %t.2.o then link with %t.main.o

// RUN: %clang -v --target=x86_64-unknown-linux-gnu \
// RUN:   --hip-link -fgpu-rdc --offload-arch=gfx906 \
// RUN:   --emit-static-lib -fuse-ld=lld -nostdlib %t.1.o %t.2.o -o %t.a \
// RUN:   2>&1 | FileCheck -check-prefix=STATIC %s
// STATIC: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID1:[0-9a-f]+]]
// STATIC: Found undefined HIP fatbin symbol: __hip_fatbin_[[ID2:[0-9a-f]+]]
// STATIC: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID1]]
// STATIC: Found undefined HIP gpubin handle symbol: __hip_gpubin_handle_[[ID2]]
// STATIC: "{{.*}}/clang-offload-bundler" {{.*}}-unbundle
// STATIC: "{{.*}}/lld" -flavor gnu -m elf64_amdgpu
// STATIC: "{{.*}}/clang-offload-bundler"
// STATIC: "{{.*}}/llvm-mc" -triple x86_64-unknown-linux-gnu
// STATIC: "{{.*}}/llvm-ar"

// RUN: %clang -v --target=x86_64-unknown-linux-gnu \
// RUN:   --hip-link -no-hip-rt -fgpu-rdc --offload-arch=gfx906 \
// RUN:   -fuse-ld=lld -nostdlib -r %t.main.o %t.a -o %t.final.o \
// RUN:   2>&1 | FileCheck -check-prefix=LINK-A %s
// LINK-A-NOT: Found undefined HIP {{.*}}symbol

#include "hip.h"

#ifdef LIB
__device__ int x;
__device__ void libfun() {
  x = 1;
}
#elif !defined(MAIN)
__device__ void libfun();
__global__ void kern() {
  libfun();
}
void run() {
  kern<<<1,1>>>();
}
#else
extern void run();
int main() {
  run();
}
#endif
