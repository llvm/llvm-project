! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -o - %s | FileCheck %s

! ------------------------------------------------------------------------------
! GENERIC KERNELS
! ------------------------------------------------------------------------------

! CHECK-LABEL: func.func @{{.*}}generic
subroutine generic(n)
  implicit none
  integer, intent(in) :: n
  integer :: i, j

  ! CHECK: omp.target kernel_type(generic)
  !$omp target
    call foo()
  !$omp end target

  ! CHECK: omp.target kernel_type(generic)
  !$omp target teams
    call foo()
  !$omp end target teams

  ! CHECK: omp.target kernel_type(generic)
  !$omp target teams distribute
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(generic)
  !$omp target teams distribute
  do i = 1, n
    !$omp parallel do
    do j = 1, n
      call foo()
    end do
  end do

  ! CHECK: omp.target kernel_type(generic)
  !$omp target teams
  !$omp parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(generic)
  !$omp target teams
  !$omp parallel
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end parallel
  !$omp end target teams
end subroutine

! ------------------------------------------------------------------------------
! BARE KERNELS
! ------------------------------------------------------------------------------

! CHECK-LABEL: func.func @{{.*}}bare
subroutine bare(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(bare)
  !$omp target teams ompx_bare num_teams(1) thread_limit(1)
    call foo()
  !$omp end target teams

  ! CHECK: omp.target kernel_type(bare)
  !$omp target teams ompx_bare num_teams(1) thread_limit(1)
  !$omp distribute parallel do simd
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(bare)
  !$omp target teams ompx_bare num_teams(1) thread_limit(1)
  !$omp distribute parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(bare)
  !$omp target teams ompx_bare num_teams(1) thread_limit(1)
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end target teams
end subroutine

! ------------------------------------------------------------------------------
! SPMD KERNELS PROMOTABLE TO NO-LOOP MODE
! ------------------------------------------------------------------------------

! CHECK-LABEL: func.func @{{.*}}target_teams_distribute_parallel_do_simd
subroutine target_teams_distribute_parallel_do_simd(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target teams distribute parallel do simd
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams distribute parallel do simd
  do i = 1, n
    call foo()
  end do
  !$omp end target

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd
  do i = 1, n
    call foo()
  end do
  !$omp end teams
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @{{.*}}target_teams_distribute_parallel_do
subroutine target_teams_distribute_parallel_do(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target teams distribute parallel do
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams distribute parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end target

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end teams
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @{{.*}}target_teams_loop
subroutine target_teams_loop(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target teams loop
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams loop
  do i = 1, n
    call foo()
  end do
  !$omp end target

  !$omp target teams
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end target teams

  ! CHECK: omp.target kernel_type(spmd_no_loop)
  !$omp target
  !$omp teams
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end teams
  !$omp end target
end subroutine

! ------------------------------------------------------------------------------
! SPMD KERNELS NOT PROMOTABLE TO NO-LOOP MODE
! ------------------------------------------------------------------------------

! CHECK-LABEL: func.func @{{.*}}target_parallel_do_simd
subroutine target_parallel_do_simd(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel do simd
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel do simd
  do i = 1, n
    call foo()
  end do
  !$omp end target

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel
  !$omp do simd
  do i = 1, n
    call foo()
  end do
  !$omp end target parallel

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel
  !$omp do simd
  do i = 1, n
    call foo()
  end do
  !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @{{.*}}target_parallel_do
subroutine target_parallel_do(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel do
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel do
  do i = 1, n
    call foo()
  end do
  !$omp end target

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel
  !$omp do
  do i = 1, n
    call foo()
  end do
  !$omp end target parallel

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel
  !$omp do
  do i = 1, n
    call foo()
  end do
  !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @{{.*}}target_parallel_loop
subroutine target_parallel_loop(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel loop
  do i = 1, n
    call foo()
  end do

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel loop
  do i = 1, n
    call foo()
  end do
  !$omp end target

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target parallel
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end target parallel

  ! CHECK: omp.target kernel_type(spmd)
  !$omp target
  !$omp parallel
  !$omp loop
  do i = 1, n
    call foo()
  end do
  !$omp end parallel
  !$omp end target
end subroutine
