// Tests the driver when linking LLVM IR bitcode files and targeting SPIR-V
// architecture.
//
// REQUIRES: spirv-registered-target
//
// Test that -Xlinker options are being passed to clang-sycl-linker.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64 --sycl-link -Xlinker --test-arg-1 -Xlinker --test-arg-2=value1,value2 %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=XLINKEROPTS
// XLINKEROPTS: "{{.*}}clang-sycl-linker{{.*}}" "--test-arg-1" "--test-arg-2=value1,value2" "{{.*}}.bc" "-o" "a.out"

// Test that -v is forwarded to clang-sycl-linker when --sycl-link is used.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64 --sycl-link -v %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=VERBOSE
// VERBOSE: "{{.*}}clang-sycl-linker{{.*}}" {{.*}}"-v"
