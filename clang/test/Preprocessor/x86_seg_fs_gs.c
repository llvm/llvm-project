// RUN: %clang -target i386-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s
// RUN: %clang -target x86_64-unknown-unknown -x c -E -dM -o - %s | FileCheck -match-full-lines %s

// CHECK: #define __SEG_FS 1
// CHECK: #define __SEG_GS 1
// CHECK: #define __seg_fs __attribute__((address_space(257)))
// CHECK: #define __seg_gs __attribute__((address_space(256)))
