! RUN: %S/../test_errors.sh %s %t %flang -fopenacc
! REQUIRES: shell

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

  !$acc cache(a(i))
  !$acc cache(a(1:2,3:4))
  !$acc cache(a)
  !$acc cache(readonly: a, aa)
  !$acc cache(readonly: a(i), aa(i, i))
  !$acc cache(t%arr)
  !$acc cache(ta(1:2)%arr)
  !$acc cache(ta(1:2)%arr(1:4))

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(ta(1:2)%s)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(i)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(t%s)

  !ERROR: Only array element or subarray are allowed in CACHE directive
  !$acc cache(/i/)

end program openacc_cache_validity
