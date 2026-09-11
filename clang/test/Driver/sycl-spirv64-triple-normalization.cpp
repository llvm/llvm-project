// Verify that --offload-targets=spirv64 (arch-only spelling) is normalized to
// the canonical spirv64-unknown-unknown triple in the offload image metadata.
// The SYCL runtime does an exact string match against "spirv64-unknown-unknown"
// so using the arch-only form causes "No compatible image found" at runtime.
//
// RUN: %clang -### -fsycl --offload-targets=spirv64 \
// RUN:   --no-offloadlib -c %s 2>&1 | FileCheck %s
// RUN: %clang_cl -### -fsycl --offload-targets=spirv64 \
// RUN:   --no-offloadlib -c -- %s 2>&1 | FileCheck %s
//
// CHECK: "--image=file={{.*}},triple=spirv64-unknown-unknown,arch=generic,kind=sycl"
