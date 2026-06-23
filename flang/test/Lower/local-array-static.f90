! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

subroutine small_local
  integer :: arr(8)
! CHECK-LABEL: func @_QPsmall_local
! CHECK: fir.alloca {{.*}}uniq_name = "_QFsmall_localEarr"
end subroutine

recursive subroutine recursive_local
  integer :: arr(8)
! CHECK-LABEL: func @_QPrecursive_local
! CHECK: fir.alloca {{.*}}uniq_name = "_QFrecursive_localEarr"
end subroutine

subroutine huge_local
  ! One byte over the 2 GiB threshold (2**31, x86-64 small code model limit).
  integer(8), parameter :: big = ishft(1_8, 31) + 1
  integer(1) :: x(big)
! CHECK-LABEL: fir.global internal @_QFhuge_localEx
! CHECK-NOT: fir.alloca {{.*}}uniq_name = "_QFhuge_localEx"
end subroutine
