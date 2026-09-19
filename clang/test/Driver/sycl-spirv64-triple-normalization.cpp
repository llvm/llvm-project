// Verify that short --offload-targets= spellings are expanded to the canonical
// spirv64-unknown-unknown triple in the offload image metadata. The SYCL
// runtime does an exact string match against "spirv64-unknown-unknown", so a
// short form causes "No compatible image found" at runtime.
//
// RUN: %clang -### -fsycl --offload-targets=spirv64 \
// RUN:   --no-offloadlib -c %s 2>&1 | FileCheck %s
// RUN: %clang_cl -### -fsycl --offload-targets=spirv64 \
// RUN:   --no-offloadlib -c -- %s 2>&1 | FileCheck %s
// RUN: %clang -### -fsycl --offload-targets=spirv64-unknown \
// RUN:   --no-offloadlib -c %s 2>&1 | FileCheck %s
//
// CHECK: "--image=file={{.*}},triple=spirv64-unknown-unknown,arch=generic,kind=sycl"

// A triple that names a vendor or an OS is used as given.
//
// RUN: %clang -### -fsycl --offload-targets=spirv64-intel \
// RUN:   --no-offloadlib -c %s 2>&1 | FileCheck --check-prefix=INTEL %s
//
// INTEL: "--image=file={{.*}},triple=spirv64-intel,arch=generic,kind=sycl"
