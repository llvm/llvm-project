! Test that $dir loop directives (known or unknown) are not clashing
! with $acc lowering.

! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test_before_acc_loop(a, b, c)
  real, dimension(10) :: a,b,c
  !dir$ myloop_directive_1
  !dir$ myloop_directive_2
  !$acc loop
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_before_acc_loop
! CHECK: acc.loop

subroutine test_after_acc_loop(a, b, c)
  real, dimension(10) :: a,b,c
  !$acc loop
  !dir$ myloop_directive_1
  !dir$ myloop_directive_2
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_after_acc_loop
! CHECK: acc.loop

subroutine test_before_acc_combined(a, b, c)
  real, dimension(10) :: a,b,c
  !dir$ myloop_directive_1
  !dir$ myloop_directive_2
  !$acc parallel loop
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_before_acc_combined
! CHECK: acc.parallel combined(loop)

subroutine test_after_acc_combined(a, b, c)
  real, dimension(10) :: a,b,c
  !$acc parallel loop
  !dir$ myloop_directive_1
  !dir$ myloop_directive_2
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_after_acc_combined
! CHECK: acc.parallel combined(loop)


subroutine test_vector_always_after_acc(a, b, c)
  real, dimension(10) :: a,b,c
  !$acc loop
  !dir$ vector always
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_vector_always_after_acc
! CHECK: acc.loop

subroutine test_vector_always_before_acc(a, b, c)
  real, dimension(10) :: a,b,c
  !dir$ vector always
  !$acc loop
  do i=1,N
    a(i) = b(i) + c(i)
  enddo
end subroutine
! CHECK-LABEL: test_vector_always_before_acc
! CHECK: acc.loop
