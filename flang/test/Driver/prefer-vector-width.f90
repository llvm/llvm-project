! Test that -mprefer-vector-width works as expected.

! RUN: %flang_fc1 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-DEF
! RUN: %flang_fc1 -mprefer-vector-width=none -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NONE
! RUN: %flang_fc1 -mprefer-vector-width=128 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-128
! RUN: %flang_fc1 -mprefer-vector-width=256 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-256
! RUN: not %flang_fc1 -mprefer-vector-width=xxx  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-INVALID

subroutine func
end subroutine func

! CHECK-DEF-NOT: attributes #0 = { "prefer-vector-width"={{.*}}
! CHECK-NONE: attributes #0 = { "prefer-vector-width"="none"
! CHECK-128: attributes #0 = { "prefer-vector-width"="128"
! CHECK-256: attributes #0 = { "prefer-vector-width"="256"
! CHECK-INVALID:error: invalid value 'xxx' in '-mprefer-vector-width=xxx'
