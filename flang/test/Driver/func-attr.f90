! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=none -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONEFP
! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=non-leaf -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-NONLEAFFP
! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=all -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ALLFP

! CHECK-LABEL: @func_() #0
subroutine func
end subroutine func

! CHECK-NONEFP: attributes #0 = { "frame-pointer"="none" }
! CHECK-NONLEAFFP: attributes #0 = { "frame-pointer"="non-leaf" }
! CHECK-ALLFP: attributes #0 = { "frame-pointer"="all" }

