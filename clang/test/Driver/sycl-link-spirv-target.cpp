// Tests the driver when linking LLVM IR bitcode files and targeting SPIR-V
// architecture.
//
// Test that -Xlinker options are being passed to clang-sycl-linker.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64 --sycl-link -Xlinker --llvm-spirv-path=/tmp \
// RUN:   -Xlinker --library-path=/tmp -Xlinker --device-libs=lib1.bc,lib2.bc %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=XLINKEROPTS
// XLINKEROPTS: "{{.*}}clang-sycl-linker{{.*}}" "--llvm-spirv-path=/tmp" "--library-path=/tmp" "--device-libs=lib1.bc,lib2.bc" "{{.*}}.bc" "-o" "a.out"
