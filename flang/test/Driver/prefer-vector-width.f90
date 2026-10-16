! Test that -mprefer-vector-width works as expected.

! RUN: %flang_fc1 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-DEF
! RUN: %flang_fc1 -mprefer-vector-width=none -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK,CHECK-NONE
! RUN: %flang_fc1 -mprefer-vector-width=128 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK,CHECK-128
! RUN: %flang_fc1 -mprefer-vector-width=256 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK,CHECK-256
! RUN: not %flang_fc1 -mprefer-vector-width=xxx  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-INVALID

subroutine func
end subroutine func

! CHECK: define {{.+}} @func{{.*}} #[[ATTRS:[0-9]+]]
! CHECK: attributes #[[ATTRS]] =
! CHECK-DEF-NOT: "prefer-vector-width"
! CHECK-NONE-SAME: "prefer-vector-width"="none"
! CHECK-128-SAME: "prefer-vector-width"="128"
! CHECK-256-SAME: "prefer-vector-width"="256"
! CHECK-INVALID: error: invalid value 'xxx' in 'mprefer-vector-width='
