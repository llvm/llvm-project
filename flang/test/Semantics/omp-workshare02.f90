! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.4 workshare Construct
! The !omp workshare construct must not contain any user defined
! function calls unless the function is ELEMENTAL.

module my_mod
  contains
  function my_func(n)
    integer :: my_func(n, n)
    my_func = 10
  end function my_func
end module my_mod

subroutine workshare(aa, bb, cc, dd, ee, ff, n)
  use my_mod
  integer n, i
  real aa(n,n), bb(n,n), cc(n,n), dd(n,n), ee(n,n), ff(n,n)

  !$omp workshare
  !ERROR: Non-ELEMENTAL function is not allowed in !$omp workshare construct
  aa = my_func(n)
  cc = dd
  ee = ff
  !$omp end workshare

end subroutine workshare
