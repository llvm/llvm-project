! UNSUPPORTED: offload-cuda

! Destruction of a zero-sized derived type component must not interpret the
! following field as an allocatable descriptor. Exercise both scalar and array
! enclosing objects, using explicit deallocation to reach the runtime without
! depending on structure constructor temporary cleanup.

! RUN: %flang %isysroot -L"%libdir" %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

program derived_array_destruction
  implicit none

  type leaf
    integer, allocatable :: payload(:)
  end type

  type outer
    type(leaf) :: part(0)
    integer(8) :: tail = 42
  end type

  type(outer), allocatable :: scalar, array(:)

  allocate(scalar)
  deallocate(scalar)

  allocate(array(2))
  deallocate(array)
end program
