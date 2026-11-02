! Test that prologue_end is placed on the first executable statement.

! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -debug-info-kind=standalone -S %s -o - | FileCheck %s

subroutine adjustable(a, n)
  integer :: n
  integer :: a(4, n+1)
  a(1,1) = 1
! CHECK-LABEL: adjustable_:
! CHECK: .loc {{[0-9]+}} [[# @LINE-2]] {{[0-9]+}} prologue_end
end subroutine

subroutine automatic(n)
  integer :: n
  real :: work(n)
  work(1) = 1.0
! CHECK-LABEL: automatic_:
! CHECK: .loc {{[0-9]+}} [[# @LINE-2]] {{[0-9]+}} prologue_end
end subroutine

subroutine assumed_len(s)
  character(*) :: s
  s(1:1) = 'x'
! CHECK-LABEL: assumed_len_:
! CHECK: .loc {{[0-9]+}} [[# @LINE-2]] {{[0-9]+}} prologue_end
end subroutine

subroutine target_array(a, n)
  integer :: n
  integer, target :: a(n)
  integer, pointer :: q(:)
  q => a
  q(1) = 1
! CHECK-LABEL: target_array_:
! CHECK: .loc {{[0-9]+}} [[# @LINE-3]] {{[0-9]+}} prologue_end
end subroutine

subroutine alt_return(a, n, *)
  integer :: n
  integer :: a(n)
  a(1) = 1
  return 1
! CHECK-LABEL: alt_return_:
! CHECK: .loc {{[0-9]+}} [[# @LINE-3]] {{[0-9]+}} prologue_end
end subroutine
