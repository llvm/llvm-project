! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of IS_CONTIGUOUS on simply contiguous items (9.5.4)
! When IS_CONTIGUOUS() is constant, it's .TRUE.

module m
  real, target :: hosted(2)
 contains
  function f()
    real, pointer, contiguous :: f(:)
    f => hosted
  end function
  subroutine test(arr1, arr2, arr3, mat, alloc)
    real, intent(in) :: arr1(:), arr2(10), mat(10, 10)
    real, intent(in), contiguous :: arr3(:)
    real, allocatable :: alloc(:)
    real :: scalar
    logical, parameter :: test_isc01 = is_contiguous(0)
    logical, parameter :: test_isc02 = is_contiguous(scalar)
    logical, parameter :: test_isc03 = is_contiguous(scalar + scalar)
    logical, parameter :: test_isc04 = is_contiguous([0, 1, 2])
    logical, parameter :: test_isc05 = is_contiguous(arr1 + 1.0)
    logical, parameter :: test_isc06 = is_contiguous(arr2)
    logical, parameter :: test_isc07 = is_contiguous(mat)
    logical, parameter :: test_isc08 = is_contiguous(mat(1:10,1))
    logical, parameter :: test_isc09 = is_contiguous(arr2(1:10:1))
    logical, parameter :: test_isc10 = is_contiguous(arr3)
    logical, parameter :: test_isc11 = is_contiguous(f())
    logical, parameter :: test_isc12 = is_contiguous(alloc)
  end subroutine
end module
