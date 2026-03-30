! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

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
! CHECK: acc.loop combined(parallel) private(%{{.*}} : !fir.ref<i32>) {
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.yield
! CHECK: acc.yield
! CHECK: } attributes {independent = [#acc.device_type<none>], unstructured}
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

! Test that acc.data is still created when there are no data clauses but the
! construct contains unstructured control flow. Without this, the early return
! in genACCDataOp skips acc.data creation, leaving orphaned blocks.
subroutine test_unstructured4(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) stop 'unstructured'
    end do
  end do
  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured4
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.terminator
! CHECK: }

! Test that GOTO exiting acc.data (one level) generates acc.terminator
! instead of an invalid cross-region branch.
subroutine test_unstructured5(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  !$acc end data
999 continue

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured5
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br

! Test that GOTO exiting acc.loop (one level) generates acc.yield
! instead of an invalid cross-region branch.
subroutine test_unstructured6(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
20  B(i) = A(i)
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured6
! CHECK: acc.loop gang vector
! CHECK: acc.loop
! CHECK: arith.cmpf ogt
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: } attributes {seq = [#acc.device_type<none>], unstructured}

! Test GOTO exiting acc.loop with intermediate code between loop end and
! target. A jump table (exit selector + dispatch) skips the intermediate code.
subroutine test_unstructured7(A, B, C, N)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*), C(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
    C(i) = 999.0
20  B(i) = A(i)
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured7
! CHECK: acc.loop gang vector
! Inner loop stores exit selector and yields:
! CHECK: acc.loop
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: } attributes {seq = [#acc.device_type<none>], unstructured}
! Jump table after inner loop:
! CHECK: fir.load %{{.*}} : !fir.ref<i32>
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br
! Intermediate code on fall-through path:
! CHECK: arith.constant 9.990000e+02

! Test GOTO exiting acc.data with intermediate code. Jump table dispatches
! after the acc.data op.
subroutine test_unstructured8(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu
  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  a(1) = -1.0
  !$acc end data
999 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured8
! Inside acc.data, GOTO stores exit selector and terminates:
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! Jump table after acc.data:
! CHECK: fir.load %{{.*}} : !fir.ref<i32>
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br
