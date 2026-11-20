! UNSUPPORTED: offload-cuda

! Fixes: #240196
! [flang] ALLOCATE with STAT= reports success when the allocation size overflows
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!

! CHECK: PASS

program alloc_overflow
  use iso_fortran_env
  implicit none

  type :: small_type
    integer(int64) :: i64
    integer(int32) :: i32
  end type

  type :: large_type
    integer(int64), dimension(1024) :: i64
    integer(int32), dimension(1024) :: i32
  end type

  type(small_type), dimension(:),   allocatable :: small
  type(large_type), dimension(:),   allocatable :: large
  type(large_type), dimension(:,:), allocatable :: large2d

  type alloc_stat
    integer(int64) :: size
    integer(int64) :: stat
    logical :: is_alloc
  end type

  type(alloc_stat) :: alloc_small
  type(alloc_stat) :: alloc_large
  type(alloc_stat) :: alloc_large2d

  alloc_small%size = 2_int64**19
  alloc_large%size = 2_int64**60
  alloc_large2d%size = 2_int64**30

  allocate(small(alloc_small%size), stat=alloc_small%stat)
  alloc_small%is_alloc = allocated(small)
  allocate(large(alloc_large%size), stat=alloc_large%stat)
  alloc_large%is_alloc = allocated(large)
  allocate(large2d(alloc_large2d%size,alloc_large2d%size), stat=alloc_large2d%stat)
  alloc_large2d%is_alloc = allocated(large2d)

  print*, 'alloc_small: size', alloc_small%size, &
    'stat', alloc_small%stat, &
    'is_alloc', alloc_small%is_alloc
  print*, 'alloc_large: size', alloc_large%size, &
    'stat', alloc_large%stat, &
    'is_alloc', alloc_large%is_alloc
  print*, 'alloc_large2d: size', alloc_large2d%size, &
    'stat', alloc_large2d%stat, &
    'is_alloc', alloc_large2d%is_alloc

  if ((alloc_small%stat == 0) .and.      &
      (alloc_small%is_alloc)  .and.      &
      (alloc_large%stat /= 0) .and.      &
      (.not. alloc_large%is_alloc) .and. &
      (alloc_large2d%stat /= 0) .and.    &
      (.not. alloc_large2d%is_alloc)) then
    print*, 'PASS'
  else
    print*, 'FAIL'
  endif

end program alloc_overflow
