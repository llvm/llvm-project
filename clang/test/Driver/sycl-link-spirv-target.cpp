// Tests the driver when linking LLVM IR bitcode files and targeting SPIR-V
// architecture.
//
// RUN: touch %t.bc
// RUN: %clangxx --target=spirv64 --sycl-link -### %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=LINK
// LINK: "{{.*}}clang-sycl-linker{{.*}}" "{{.*}}.bc" "-o" "a.out"
