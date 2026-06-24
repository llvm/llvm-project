! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: func @_QPtest_ierrno(
subroutine test_ierrno()
    integer :: i
    i = ierrno()
! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_ierrnoEi"}
! CHECK: %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFtest_ierrnoEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK: %[[VAL_2:.*]] = fir.call @_QPierrno() fastmath<contract> : () -> i32
! CHECK: fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK: return
end subroutine test_ierrno
