! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! XFAIL: *

subroutine test_unstructured1(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc data copy(a, b, c)

  !$acc kernels
  a(:,:,:) = 0.0
  !$acc end kernels

  !$acc kernels
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do
  end do
  !$acc end kernels

  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do

    if (a(1,2,3) > 10) stop 'just to be unstructured'
  end do

  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured1
! CHECK: acc.data
! CHECK: acc.kernels
! CHECK: acc.kernels
! CHECK: fir.call @_FortranAStopStatementText


subroutine test_unstructured2(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel loop
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do

! CHECK-LABEL: func.func @_QPtest_unstructured2
! CHECK: acc.parallel
! CHECK: acc.loop
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.yield
! CHECK: acc.yield
! CHECK: acc.yield

end subroutine

subroutine test_unstructured3(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
  !$acc end parallel

! CHECK-LABEL: func.func @_QPtest_unstructured3
! CHECK: acc.parallel
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.yield
! CHECK: acc.yield

end subroutine
