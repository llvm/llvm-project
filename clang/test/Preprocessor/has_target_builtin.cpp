// RUN: %clang_cc1 -fopenmp -triple=spirv64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=HAS %s

// RUN: %clang_cc1 -fopenmp -triple=nvptx64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=HAS %s

// RUN: %clang_cc1 -fopenmp -triple=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=HAS %s

// RUN: %clang_cc1 -fopenmp -triple=aarch64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=HAS %s

// RUN: %clang_cc1 -triple=aarch64 -E %s | FileCheck -implicit-check-not=HAS %s

// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -E %s | \
// RUN: FileCheck -check-prefix=CHECK-NO-OFFLOAD-HAS-BUILTIN -implicit-check-not=DOESNT %s

// CHECK: DOESNT
// CHECK-NO-OFFLOAD-HAS-BUILTIN: HAS
#if __has_target_builtin(__builtin_ia32_pause)
 HAS
#else
 DOESNT
#endif
