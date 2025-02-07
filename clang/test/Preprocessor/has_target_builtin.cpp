// RUN: %clang_cc1 -fopenmp -triple=spirv64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=BAD %s

// RUN: %clang_cc1 -fopenmp -triple=nvptx64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=BAD %s

// RUN: %clang_cc1 -fopenmp -triple=amdgcn-amd-amdhsa -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=BAD %s

// RUN: %clang_cc1 -fopenmp -triple=aarch64 -fopenmp-is-target-device \
// RUN: -aux-triple x86_64-linux-unknown -E %s | FileCheck -implicit-check-not=BAD %s

// CHECK: GOOD
#if __has_target_builtin(__builtin_ia32_pause)
  BAD
#else
  GOOD
#endif
