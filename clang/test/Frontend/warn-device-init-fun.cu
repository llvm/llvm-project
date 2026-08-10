// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:     -fgpu-allow-device-init \
// RUN:      %s 2>&1 | FileCheck %s

// CHECK: warning: '-fgpu-allow-device-init' is ignored since it is only supported for HIP

