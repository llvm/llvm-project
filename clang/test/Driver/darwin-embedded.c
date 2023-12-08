// RUN: %clang -target x86_64-apple-darwin -arch armv6m -resource-dir=%S/Inputs/resource_dir %s -### 2> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -mfloat-abi=soft -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t

// RUN: %clang -target x86_64-apple-darwin -arch armv7m -fPIC -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -fPIC -mfloat-abi=hard -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -fPIC -mfloat-abi=softfp -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-none-macho -arch armv7 -mhard-float -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t
// RUN: %clang -target x86_64-apple-none-macho -arch armv7 -msoft-float -fPIC -resource-dir=%S/Inputs/resource_dir %s -### 2>> %t


// RUN: FileCheck %s < %t

// ARMv6m has no float
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "soft"
// CHECK: libclang_rt.soft_static.a

// ARMv7em does
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "hard"
// CHECK: libclang_rt.hard_static.a

// but the ABI can be overridden
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-target-feature" "+soft-float"
// CHECK: "-mfloat-abi" "soft"
// CHECK: libclang_rt.soft_static.a

// ARMv7m has no float either
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "soft"
// CHECK: libclang_rt.soft_pic.a

// But it can be enabled on ARMv7em
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "hard"
// CHECK: libclang_rt.hard_pic.a

// "softfp" must link against a soft-float library since that's what the
// callers we're compiling will expect.
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "soft"
// CHECK: libclang_rt.soft_pic.a

// -arch "armv7" (== embedded v7a) can be used in a couple of variants:
// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "hard"
// CHECK: libclang_rt.hard_static.a

// CHECK-LABEL: Target:
// CHECK-NOT: warning: unknown platform
// CHECK: "-mfloat-abi" "soft"
// CHECK: libclang_rt.soft_pic.a
