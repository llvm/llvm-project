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

// Test that -triple is propagated from --target and passed to clang-sycl-linker.
// Test that no -march= results in no -arch= in clang-sycl-linker command line.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64-unknown-unknown --sycl-link %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FINALIZE
// FINALIZE: "{{.*}}clang-sycl-linker{{.*}}" "{{.*}}.bc" "-o" "a.out" "-triple=spirv64-unknown-unknown"{{$}}

// Test that the target triple is passed on as spelled rather than padded out
// to a full triple.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64 --sycl-link %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FINALIZE-SHORT
// FINALIZE-SHORT: "{{.*}}clang-sycl-linker{{.*}}" "{{.*}}.bc" "-o" "a.out" "-triple=spirv64"{{$}}

// Test that a requested device architecture is passed on as -arch=.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64-unknown-unknown -march=foo --sycl-link %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=FINALIZE-ARCH
// FINALIZE-ARCH: "{{.*}}clang-sycl-linker{{.*}}" "{{.*}}.bc" "-o" "a.out" "-triple=spirv64-unknown-unknown" "-arch=foo"{{$}}

// Test that -triple=/-arch= passed through -Xlinker/-Wl take priority and that
// --target/-march passed to clang do not cause duplication of -triple=/-arch=.
// RUN: touch %t.bc
// RUN: %clangxx -### --target=spirv64 --sycl-link -march=bmg_g21 -Xlinker -triple=spirv64-unknown-unknown -Xlinker -arch=bar %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NODUP
// RUN: %clangxx -### --target=spirv64 --sycl-link -march=bmg_g21 -Wl,-triple=spirv64-unknown-unknown,-arch=bar %t.bc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NODUP
// NODUP: "{{.*}}clang-sycl-linker{{.*}}" "-triple=spirv64-unknown-unknown" "-arch=bar" "{{.*}}.bc" "-o" "a.out"{{$}}
