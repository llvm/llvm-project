// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=128 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=128
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=256 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=512 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=512
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=1024 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=1024
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=2048 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=2048
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -msve-vector-bits=128 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=128
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -msve-vector-bits=256 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=256
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -msve-vector-bits=scalable -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -msve-vector-bits=scalable -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONE

// CHECK-LABEL: @func() #0
// CHECK: attributes #0 = { {{.*}} vscale_range([[#div(VBITS,128)]],[[#div(VBITS,128)]]) {{.*}} }
// CHECK-NONE: attributes #0 = { {{.*}} vscale_range(0,16) {{.*}} }
void func() {}
