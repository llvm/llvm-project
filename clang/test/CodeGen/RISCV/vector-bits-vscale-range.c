// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -mvscale-min=1 -mvscale-max=1 -emit-llvm -o - %s | FileCheck %s -D#VBITS=1
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -mvscale-max=2 -emit-llvm -o - %s | FileCheck %s -D#VBITS=2
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -mvscale-max=4 -emit-llvm -o - %s | FileCheck %s -D#VBITS=4
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -mvscale-max=8 -emit-llvm -o - %s | FileCheck %s -D#VBITS=8
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -mvscale-max=16 -emit-llvm -o - %s | FileCheck %s -D#VBITS=16
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -mvscale-min=1 -emit-llvm -o - %s | FileCheck %s -D#VBITS=1 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -emit-llvm -o - %s | FileCheck %s -D#VBITS=4 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -emit-llvm -o - %s | FileCheck %s -D#VBITS=8 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -emit-llvm -o - %s | FileCheck %s -D#VBITS=16 --check-prefix=CHECK-NOMAX
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -mvscale-min=1 -mvscale-max=0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-UNBOUNDED
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-V
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v -target-feature +zvl512b -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ZVL
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ZVE64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64f -target-feature +f -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ZVE64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve64d -target-feature +f -target-feature +d -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ZVE64
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +zve32x -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ZVE32

// CHECK-LABEL: @func() #0
// CHECK: attributes #0 = { {{.*}} vscale_range([[#VBITS]],[[#VBITS]]) {{.*}} }
// CHECK-NOMAX: attributes #0 = { {{.*}} vscale_range([[#VBITS]],0) {{.*}} }
// CHECK-UNBOUNDED: attributes #0 = { {{.*}} vscale_range(1,0) {{.*}} }
// CHECK-V: attributes #0 = { {{.*}} vscale_range(2,1024) {{.*}} }
// CHECK-ZVL: attributes #0 = { {{.*}} vscale_range(8,1024) {{.*}} }
// CHECK-ZVE64: attributes #0 = { {{.*}} vscale_range(1,1024) {{.*}} }
// CHECK-ZVE32: attributes #0
// CHECK-ZVE32-NOT: vscale_range
void func(void) {}
