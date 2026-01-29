// Test HIPSPV static device library linking
// REQUIRES: system-linux
// UNSUPPORTED: system-windows

// Create a dummy archive to test SDL linking
// RUN: rm -rf %t && mkdir %t
// RUN: touch %t/dummy.bc
// RUN: llvm-ar cr %t/libSDL.a %t/dummy.bc

// Test that -l options are passed to llvm-link for --offload=spirv64
// RUN: %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN:   --no-offload-new-driver -L%t -lSDL \
// RUN: 2>&1 | FileCheck -check-prefixes=ALL,SDL %s

// Test that .a files are properly unbundled and passed to llvm-link
// RUN: %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN:   --no-offload-new-driver %t/libSDL.a \
// RUN: 2>&1 | FileCheck -check-prefixes=ALL,SDL %s

// RUN: %clang --target=x86_64-linux-gnu --offload-new-driver -fgpu-rdc \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   -nogpuinc -nogpulib %s -c -o %t/tu0.o
// RUN: %clang --target=x86_64-linux-gnu --offload-new-driver -fgpu-rdc \
// RUN:   --offload=spirv64-unknown-chipstar \
// RUN:   -nogpuinc -nogpulib %s -c -o %t/tu1.o
// RUN: llvm-ar cr %t/libSDL2.a %t/tu1.o

// RUN: %clang --target=x86_64-linux-gnu --offload-new-driver -fgpu-rdc \
// RUN:   -Xoffload-compiler-spirv64-unknown-chipstar \
// RUN:   --hip-path=%S/Inputs/hipspv -no-hip-rt %t/tu0.o %t/libSDL2.a \
// A hacky trick to print clang commands invoked by clang-linker-wrapper
// RUN:   -v -Xlinker --emit-fatbin-only -Xoffload-compiler '-###' \
// RUN: 2>&1 | FileCheck -check-prefixes=ALL,SDL-NEW %s

// Verify that the input files are added before the SDL files in llvm-link command
// This tests the ordering fix to match HIPAMD behavior
// SDL: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libSDL.a" "-targets=hip-spirv64-unknown-unknown-unknown-generic" "-output=[[SDL_A:.*\.a]]" "-allow-missing-bundles"
// SDL: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.bc" "[[SDL_A]]"
// SDL-NEW: "{{.*}}clang-linker-wrapper"
// SDL-NEW: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.o" "{{.*}}.o"

// ALL: "{{.*}}opt"
// ALL-SAME: "-load-pass-plugin" {{".*/hipspv/lib/libLLVMHipSpvPasses.so"}}
// ALL-SAME: "-passes=hip-post-link-passes"
