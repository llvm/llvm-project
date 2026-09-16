// Test HIPSPV static device library linking
// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target
// UNSUPPORTED: system-windows

// RUN: rm -rf %t && mkdir %t
// RUN: %clang -cc1 %s -triple spirv64-unknown-chipstar -emit-llvm-bc -o %t/dev.bc
// RUN: llvm-offload-binary -o %t/dev.out \
// RUN:   --image=file=%t/dev.bc,kind=hip,triple=spirv64-unknown-chipstar,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t/tu0.o \
// RUN:   -fembed-offload-object=%t/dev.out
// RUN: cp %t/tu0.o %t/tu1.o
// RUN: llvm-ar cr %t/libSDL2.a %t/tu1.o

// RUN: %clang -### --target=x86_64-linux-gnu --offload-link -fgpu-rdc \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -no-hip-rt %t/tu0.o %t/libSDL2.a \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL-NEW %s -DHIP_PATH=%S/Inputs/hipspv

// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path=%S/Inputs/hipspv \
// RUN:   --linker-path=/usr/bin/ld -o a.out %t/tu0.o %t/libSDL2.a \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL-NEW-WRAPPER %s -DHIP_PATH=%S/Inputs/hipspv

// SDL-NEW: "{{.*}}clang-linker-wrapper"
// SDL-NEW-SAME: "--device-compiler=spirv64-unknown-chipstar=--hip-path=[[HIP_PATH]]"
// SDL-NEW-SAME: "{{.*}}/tu0.o" "{{.*}}/libSDL2.a"

// SDL-NEW-WRAPPER: clang{{.*}}" --no-default-config -o {{[^ ]*.img}}
// SDL-NEW-WRAPPER-SAME: --target=spirv64-unknown-chipstar
// SDL-NEW-WRAPPER-SAME: {{[^ ]*.o}}
// SDL-NEW-WRAPPER-SAME: --hip-path=[[HIP_PATH]]
