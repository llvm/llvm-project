// REQUIRES: zstd

// RUN: %clang -### --target=aarch64-unknown-linux-gnu -gz=zstd -x assembler %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=x86_64-pc-freebsd -gz=zstd %s 2>&1 | FileCheck %s

// CHECK: {{"-cc1(as)?".* "--compress-debug-sections=zstd"}}
// CHECK: "--compress-debug-sections=zstd"
