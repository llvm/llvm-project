! REQUIRES: riscv-registered-target
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=1  -emit-llvm -o - %s | FileCheck %s -D#VBITS=1
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2 -mvscale-max=2  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4 -mvscale-max=4  -emit-llvm -o - %s | FileCheck %s -D#VBITS=4
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8 -mvscale-max=8  -emit-llvm -o - %s | FileCheck %s -D#VBITS=8
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16 -mvscale-max=16  -emit-llvm -o - %s | FileCheck %s -D#VBITS=16
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1  -emit-llvm -o - %s | FileCheck %s -D#VBITS=1 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=2  -emit-llvm -o - %s | FileCheck %s -D#VBITS=2 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=4  -emit-llvm -o - %s | FileCheck %s -D#VBITS=4 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=8  -emit-llvm -o - %s | FileCheck %s -D#VBITS=8 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=16  -emit-llvm -o - %s | FileCheck %s -D#VBITS=16 --check-prefix=CHECK-NOMAX
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -mvscale-min=1 -mvscale-max=0  -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-UNBOUNDED
! RUN: %flang_fc1 -triple riscv64-none-linux-gnu -target-feature +v -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONE

! CHECK-LABEL: @func_() #0
! CHECK: attributes #0 = {{{.*}} vscale_range([[#VBITS]],[[#VBITS]]) {{.*}}}
! CHECK-NOMAX: attributes #0 = {{{.*}} vscale_range([[#VBITS]],0) {{.*}}}
! CHECK-UNBOUNDED: attributes #0 = {{{.*}} vscale_range(1,0) {{.*}}}
! CHECK-NONE-NOT: vscale_range
subroutine func
end subroutine func
