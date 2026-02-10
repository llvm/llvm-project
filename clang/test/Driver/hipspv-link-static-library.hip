// Test HIPSPV static device library linking
// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target
// UNSUPPORTED: system-windows

// Create a dummy archive to test SDL linking
// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/dummy.bc
// RUN: llvm-ar cr %t/libSDL.a %t/dummy.bc

// Test that -l options are passed to llvm-link for --offload=spirv64
// RUN: %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN:   --no-offload-new-driver -L%t -lSDL \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL %s

// Test that .a files are properly unbundled and passed to llvm-link
// RUN: %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN:   --no-offload-new-driver %t/libSDL.a \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL %s

// RUN: %clang -cc1 %s -triple spirv64-unknown-chipstar -emit-llvm-bc -o %t/dev.bc
// RUN: llvm-offload-binary -o %t/dev.out \
// RUN:   --image=file=%t/dev.bc,kind=hip,triple=spirv64-unknown-chipstar,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t/tu0.o \
// RUN:   -fembed-offload-object=%t/dev.out
// RUN: cp %t/tu0.o %t/tu1.o
// RUN: llvm-ar cr %t/libSDL2.a %t/tu1.o

// RUN: %clang -### --target=x86_64-linux-gnu --offload-new-driver -fgpu-rdc \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -no-hip-rt %t/tu0.o %t/libSDL2.a \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL-NEW %s -DHIP_PATH=%S/Inputs/hipspv

// RUN: clang-linker-wrapper --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --device-compiler=spirv64-unknown-chipstar=--hip-path=%S/Inputs/hipspv \
// RUN:   --linker-path=/usr/bin/ld -o a.out %t/tu0.o %t/libSDL2.a \
// RUN: 2>&1 | FileCheck -check-prefixes=SDL-NEW-WRAPPER %s -DHIP_PATH=%S/Inputs/hipspv

// Verify that the input files are added before the SDL files in llvm-link command
// This tests the ordering fix to match HIPAMD behavior
// SDL: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libSDL.a" "-targets=hip-spirv64-unknown-unknown-unknown-generic" "-output=[[SDL_A:.*\.a]]" "-allow-missing-bundles"
// SDL: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.bc" "[[SDL_A]]"
// SDL-NEW: "{{.*}}clang-linker-wrapper"
// SDL-NEW-SAME: "--device-compiler=spirv64-unknown-chipstar=--hip-path=[[HIP_PATH]]"
// SDL-NEW-SAME: "{{.*}}/tu0.o" "{{.*}}/libSDL2.a"
// DELETE-SDL-NEW: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.o" "{{.*}}.o"

// SDL-NEW-WRAPPER: clang{{.*}}" --no-default-config -o {{[^ ]*.img}}
// SDL-NEW-WRAPPER-SAME: {{[^ ]*.o}} {{[^ ]*.o}}
// SDL-NEW-WRAPPER-SAME: --hip-path=[[HIP_PATH]]

// SDL: "{{.*}}opt"
// SDL-SAME: "-load-pass-plugin" {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// SDL-SAME: "-passes=hip-post-link-passes"
