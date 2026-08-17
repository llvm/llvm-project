! Executable regression test for -finit-local= array initialization.
! Verifies that every element of static arrays (1-D, 2-D, 3-D) and every
! field of every element in arrays of derived types are initialized to the
! requested bit pattern.  A previous bug caused the flat loop to use an
! out-of-bounds GEP for rank > 1, so only the first row was written.
!
! UNSUPPORTED: offload-cuda
!
! RUN: %flang %isysroot -L"%libdir" -finit-local=0xAA %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t

program test_finit_local_array
  implicit none
  integer(4), parameter :: EXPECTED = int(z'AAAAAAAA')

  ! 1-D integer array x(4) -- 4 elements
  call check_1d()

  ! 2-D integer array x(3,4) -- 12 elements
  call check_2d()

  ! 3-D integer array x(2,3,4) -- 24 elements
  call check_3d()

  ! 1-D array of derived type x(2) -- 2 elements, 2 fields each
  call check_struct_1d()

  ! 2-D array of derived type x(2,3) -- 6 elements, 2 fields each
  call check_struct_2d()

contains

  subroutine check_1d()
    integer(4) :: x(4)
    integer :: i
    do i = 1, 4
      if (x(i) /= EXPECTED) then
        print *, "FAIL check_1d: element", i, "=", x(i), "expected", EXPECTED
        error stop 1
      end if
    end do
  end subroutine

  subroutine check_2d()
    integer(4) :: x(3,4)
    integer :: i, j
    do j = 1, 4
      do i = 1, 3
        if (x(i,j) /= EXPECTED) then
          print *, "FAIL check_2d: element (", i, ",", j, ")=", x(i,j), &
                   "expected", EXPECTED
          error stop 1
        end if
      end do
    end do
  end subroutine

  subroutine check_3d()
    integer(4) :: x(2,3,4)
    integer :: i, j, k
    do k = 1, 4
      do j = 1, 3
        do i = 1, 2
          if (x(i,j,k) /= EXPECTED) then
            print *, "FAIL check_3d: element (", i, ",", j, ",", k, ")=", &
                     x(i,j,k), "expected", EXPECTED
            error stop 1
          end if
        end do
      end do
    end do
  end subroutine

  subroutine check_struct_1d()
    type :: t
      integer(4) :: a
      integer(4) :: b
    end type
    type(t) :: x(2)
    integer :: i
    do i = 1, 2
      if (x(i)%a /= EXPECTED .or. x(i)%b /= EXPECTED) then
        print *, "FAIL check_struct_1d: element", i, &
                 "a=", x(i)%a, "b=", x(i)%b, "expected", EXPECTED
        error stop 1
      end if
    end do
  end subroutine

  subroutine check_struct_2d()
    type :: t
      integer(4) :: a
      integer(4) :: b
    end type
    type(t) :: x(2,3)
    integer :: i, j
    do j = 1, 3
      do i = 1, 2
        if (x(i,j)%a /= EXPECTED .or. x(i,j)%b /= EXPECTED) then
          print *, "FAIL check_struct_2d: element (", i, ",", j, &
                   ") a=", x(i,j)%a, "b=", x(i,j)%b, "expected", EXPECTED
          error stop 1
        end if
      end do
    end do
  end subroutine

end program
