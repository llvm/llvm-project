! Test that -mframe-pointer can accept only specific values and when given an invalid value, check it raises an error.

! REQUIRES: aarch64-registered-target

! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=none -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NONEFP
! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=non-leaf -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NONLEAFFP
! RUN: %flang_fc1 -triple aarch64-none-none -mframe-pointer=all -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-ALLFP
! RUN: not %flang_fc1 -triple aarch64-none-none -mframe-pointer=wrongval -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-WRONGVALUEFP

! CHECK-NONEFP-LABEL: @func_() {

! CHECK-NONLEAFFP-LABEL: @func_()
! CHECK-NONLEAFFP-SAME: #0

! CHECK-ALLFP-LABEL: @func_()
! CHECK-ALLFP-SAME: #0

subroutine func
end subroutine func

! CHECK-NONEFP-NOT: attributes #0 = { "frame-pointer"="{{.*}}" }
! CHECK-NONLEAFFP: attributes #0 = { "frame-pointer"="non-leaf" }
! CHECK-ALLFP: attributes #0 = { "frame-pointer"="all" }

! CHECK-WRONGVALUEFP:error: invalid value 'wrongval' in '-mframe-pointer=wrongval'
