! UNSUPPORTED: offload-cuda

! Issue: #240196
! [flang] ALLOCATE with STAT= reports success when the allocation size overflows
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!

! CHECK: PASS

program flang_overflow_success
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

  type(small_type), dimension(:), allocatable :: small
  type(large_type), dimension(:), allocatable :: large

  type alloc_stat
    integer(int64) :: size
    integer(int64) :: stat
    logical :: is_alloc
  end type

  type(alloc_stat) :: alloc_small
  type(alloc_stat) :: alloc_large

  alloc_small%size = 2_int64**20
  alloc_large%size = 2_int64**62

  allocate(small(alloc_small%size), stat=alloc_small%stat)
  alloc_small%is_alloc = allocated(small)
  allocate(large(alloc_large%size), stat=alloc_large%stat)
  alloc_large%is_alloc = allocated(large)

  print*, 'alloc_small: size', alloc_small%size, &
    'stat', alloc_small%stat, &
    'is_alloc', alloc_small%is_alloc
  print*, 'alloc_large: size', alloc_large%size, &
    'stat', alloc_large%stat, &
    'is_alloc', alloc_large%is_alloc

  if ((alloc_small%stat == 0) .and.  &
      (alloc_small%is_alloc)  .and.  &
      (alloc_large%stat /= 0) .and.  &
      (.not. alloc_large%is_alloc)) then
    print*, 'PASS'
  else
    print*, 'FAIL'
  endif

end program flang_overflow_success
