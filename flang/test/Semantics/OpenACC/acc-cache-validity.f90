! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.10 Cache

program openacc_cache_validity

  implicit none

  type atype
    real(8), dimension(10) :: arr
    real(8) :: s
  end type atype

  integer :: i
  integer, parameter :: N = 256
  real(8), dimension(N, N) :: aa
  type(atype) :: t
  type(atype), dimension(10) :: ta
  real(8), dimension(N) :: a

  do i = 1, N

  !$acc cache(a(i))
  !$acc cache(a(1:2,3:4))
  !$acc cache(a)
  !$acc cache(readonly: a, aa)
  !$acc cache(readonly: a(i), aa(i, i))
  !$acc cache(t%arr)
  !$acc cache(ta(1:2)%arr)
  !$acc cache(ta(1:2)%arr(1:4))
  !$acc cache(i)
  !$acc cache(t%s)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(t)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(/i/)

  !$acc cache(a(:)) ! ok

  !$acc cache(aa(:,:)) ! ok

  !ERROR: The CACHE directive does not support strided array sections
  !$acc cache(a(1:10:2))

  !ERROR: The CACHE directive does not support strided array sections
  !$acc cache(aa(1:10:2, 1:5))

  end do

  !$acc cache(a)

  call routine_with_cache()

contains

  subroutine routine_with_cache()
    real(8), dimension(N) :: local_arr
    integer :: j
    !$acc routine vector
    !$acc cache(local_arr)
    !$acc loop
    do j = 1, N
      local_arr(j) = a(j)
    end do
  end subroutine

end program openacc_cache_validity
