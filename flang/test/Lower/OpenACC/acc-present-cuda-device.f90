! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_parallel(
! CHECK: acc.deviceptr{{.*}}name = "b"
! CHECK: acc.present{{.*}}name = "a"
! CHECK: acc.delete{{.*}}name = "a"
subroutine test_parallel(a, b, n)
  real(8), intent(inout) :: a(:)
  real(8), device, intent(in) :: b(:)
  integer :: n
  !$acc parallel present(a, b)
  a(1) = b(1)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_data(
! CHECK: acc.deviceptr{{.*}}name = "b"
! CHECK: acc.present{{.*}}name = "a"
! CHECK: acc.delete{{.*}}name = "a"
subroutine test_data(a, b, n)
  real(8), intent(inout) :: a(:)
  real(8), device, intent(in) :: b(:)
  integer :: n
  !$acc data present(a, b)
  a(1) = b(1)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPtest_managed(
! CHECK: acc.present{{.*}}name = "a"
! CHECK: acc.present{{.*}}name = "b"
! CHECK: acc.delete{{.*}}name = "a"
! CHECK: acc.delete{{.*}}name = "b"
subroutine test_managed(a, b, n)
  real(8), intent(inout) :: a(:)
  real(8), managed, intent(in) :: b(:)
  integer :: n
  !$acc parallel present(a, b)
  a(1) = b(1)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_unified(
! CHECK: acc.present{{.*}}name = "a"
! CHECK: acc.present{{.*}}name = "b"
! CHECK: acc.delete{{.*}}name = "a"
! CHECK: acc.delete{{.*}}name = "b"
subroutine test_unified(a, b, n)
  real(8), intent(inout) :: a(:)
  real(8), unified, intent(in) :: b(:)
  integer :: n
  !$acc parallel present(a, b)
  a(1) = b(1)
  !$acc end parallel
end subroutine
