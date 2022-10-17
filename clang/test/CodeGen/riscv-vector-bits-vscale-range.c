// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=1 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=1
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -mvscale-max=2 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=2
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -mvscale-max=4 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=4
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -mvscale-max=8 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=8
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -mvscale-max=16 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=16
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=1 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=4 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=8 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -S -emit-llvm -o - %s | FileCheck %s -D#VBITS=16 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=0 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-UNBOUNDED
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONE

// CHECK-LABEL: @func() #0
// CHECK: attributes #0 = { {{.*}} vscale_range([[#VBITS]],[[#VBITS]]) {{.*}} }
// CHECK-NOMAX: attributes #0 = { {{.*}} vscale_range([[#VBITS]],0) {{.*}} }
// CHECK-UNBOUNDED: attributes #0 = { {{.*}} vscale_range(1,0) {{.*}} }
// CHECK-NONE: attributes #0 = { {{.*}} vscale_range(2,1024) {{.*}} }
void func(void) {}
