! RUN: %S/test_folding.sh %s %t %f18
! Test folding of IS_CONTIGUOUS on simply contiguous items (9.5.4)
! When IS_CONTIGUOUS() is constant, it's .TRUE.

module m
  real, target :: hosted(2)
 contains
  function f()
    real, pointer, contiguous :: f(:)
    f => hosted
  end function
  subroutine test(arr1, arr2, arr3, mat)
    real, intent(in) :: arr1(:), arr2(10), mat(10, 10)
    real, intent(in), contiguous :: arr3(:)
    real :: scalar
    logical, parameter :: isc01 = is_contiguous(0)
    logical, parameter :: isc02 = is_contiguous(scalar)
    logical, parameter :: isc03 = is_contiguous(scalar + scalar)
    logical, parameter :: isc04 = is_contiguous([0, 1, 2])
    logical, parameter :: isc05 = is_contiguous(arr1 + 1.0)
    logical, parameter :: isc06 = is_contiguous(arr2)
    logical, parameter :: isc07 = is_contiguous(mat)
    logical, parameter :: isc08 = is_contiguous(mat(1:10,1))
    logical, parameter :: isc09 = is_contiguous(arr2(1:10:1))
    logical, parameter :: isc10 = is_contiguous(arr3)
    logical, parameter :: isc11 = is_contiguous(f())
  end subroutine
end module
