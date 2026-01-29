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
// RUN:   -L%t -lSDL \
// RUN: 2>&1 | FileCheck -check-prefix=SDL-LINK %s

// Test that .a files are properly unbundled and passed to llvm-link  
// RUN: %clang -### --target=x86_64-linux-gnu --offload=spirv64 \
// RUN:   --hip-path=%S/Inputs/hipspv -nohipwrapperinc %s \
// RUN:   %t/libSDL.a \
// RUN: 2>&1 | FileCheck -check-prefix=SDL-ARCHIVE %s

// Verify that the input files are added before the SDL files in llvm-link command
// This tests the ordering fix to match HIPAMD behavior
// SDL-LINK: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libSDL.a" "-targets=hip-spirv64-unknown-unknown-unknown-generic" "-output=[[SDL_A:.*\.a]]" "-allow-missing-bundles"
// SDL-LINK: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.bc" "[[SDL_A]]"

// SDL-ARCHIVE: "{{.*}}clang-offload-bundler" "-unbundle" "-type=a" "-input={{.*}}libSDL.a" "-targets=hip-spirv64-unknown-unknown-unknown-generic" "-output=[[SDL_A:.*\.a]]" "-allow-missing-bundles"  
// SDL-ARCHIVE: "{{.*}}llvm-link" "-o" "{{.*}}.bc" "{{.*}}.bc" "[[SDL_A]]"
