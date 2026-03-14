! REQUIRES: riscv-registered-target
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=1  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -mvscale-max=2  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -mvscale-max=4  -emit-llvm -o - %s | FileCheck %s -D#VBITS=4
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -mvscale-max=8  -emit-llvm -o - %s | FileCheck %s -D#VBITS=8
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -mvscale-max=16  -emit-llvm -o - %s | FileCheck %s -D#VBITS=16
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4  -emit-llvm -o - %s | FileCheck %s -D#VBITS=4 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8  -emit-llvm -o - %s | FileCheck %s -D#VBITS=8 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16  -emit-llvm -o - %s | FileCheck %s -D#VBITS=16 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=0 -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-UNBOUNDED
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -target-feature +zvl64b -emit-llvm -o - %s | FileCheck %s -D#VBITS=1 -check-prefix=CHECK-ZVL
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 -check-prefix=CHECK-ZVL
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -target-feature +zvl128b -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 -check-prefix=CHECK-ZVL
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -target-feature +zvl256b -emit-llvm -o - %s | FileCheck %s -D#VBITS=4 -check-prefix=CHECK-ZVL
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -target-feature +zvl512b -emit-llvm -o - %s | FileCheck %s -D#VBITS=8 -check-prefix=CHECK-ZVL
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +zve64x -target-feature +zvl1024b -emit-llvm -o - %s | FileCheck %s -D#VBITS=16 -check-prefix=CHECK-ZVL

! CHECK-LABEL: @func_() #0
! CHECK: attributes #0 = {{{.*}} vscale_range([[#VBITS]],[[#VBITS]]) {{.*}}}
! CHECK-NOMAX: attributes #0 = {{{.*}} vscale_range([[#VBITS]],0) {{.*}}}
! CHECK-UNBOUNDED: attributes #0 = {{{.*}} vscale_range(2,0) {{.*}}}
! CHECK-ZVL: attributes #0 = {{{.*}} vscale_range([[#VBITS]],1024) {{.*}}}
subroutine func
end subroutine func
