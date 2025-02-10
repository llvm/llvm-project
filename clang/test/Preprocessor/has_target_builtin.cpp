// RUN: %clang_cc1 -fopenmp -triple=spirv64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not="{{BAD|DOESNT}}" %s

// RUN: %clang_cc1 -fopenmp -triple=nvptx64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not="{{BAD|DOESNT}}" %s

// RUN: %clang_cc1 -fopenmp -triple=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not="{{BAD|DOESNT}}" %s

// RUN: %clang_cc1 -fopenmp -triple=aarch64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not="{{BAD|DOESNT}}" %s

// RUN: %clang_cc1 -triple=aarch64 -E %s | FileCheck -check-prefix=CHECK-NOTOFFLOAD -implicit-check-not="{{GOOD|HAS|BAD}}" %s

// CHECK: GOOD

// CHECK-NOTOFFLOAD: DOESNT
#ifdef __has_target_builtin
  HAS
#if __has_target_builtin(__builtin_ia32_pause)
  BAD
#else
  GOOD
#endif
#else
  DOESNT
#endif
