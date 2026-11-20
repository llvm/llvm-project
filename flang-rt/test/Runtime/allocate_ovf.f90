! UNSUPPORTED: offload-cuda

! Fixes: #240196
! [flang] ALLOCATE with STAT= reports success when the allocation size overflows
!
! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
!

! CHECK: PASS

program alloc_overflow
  implicit none

  type :: small_type
    integer(8) :: i64
    integer(4) :: i32
  end type

  type :: large_type
    integer(8), dimension(1024) :: i64
    integer(4), dimension(1024) :: i32
  end type

  type(small_type), dimension(:),   allocatable :: small
  type(large_type), dimension(:),   allocatable :: large
  type(large_type), dimension(:,:), allocatable :: large2d

  type alloc_stat
    integer(8) :: size
    integer(8) :: stat
    logical :: is_alloc
  end type

  type(alloc_stat) :: alloc_small
  type(alloc_stat) :: alloc_large
  type(alloc_stat) :: alloc_large2d

  logical :: pass = .true.

  alloc_small%size = 2_8**19
  alloc_large%size = 2_8**60
  alloc_large2d%size = 2_8**30

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

  if (alloc_small%stat /= 0 .and. .not. alloc_small%is_alloc) then
    print*, 'FAIL - small allocation should have succeeded'
    pass = .false.
  endif

  if (alloc_large%stat == 0 .or. alloc_large%is_alloc) then
    print*, 'FAIL - larrge 1d allocation should have have failed'
    pass = .false.
  end if

  if (alloc_large2d%stat == 0 .or. alloc_large2d%is_alloc) then
    print*, 'FAIL - larrge 2d allocation should have have failed'
    pass = .false.
  end if

  if (pass) then
    print*, 'PASS'
    stop
  endif

  print*, 'FAIL'
  stop 1

end program alloc_overflow
