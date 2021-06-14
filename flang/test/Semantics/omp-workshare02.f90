! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.4 workshare Construct
! The !omp workshare construct must not contain any user defined
! function calls unless the function is ELEMENTAL.

module my_mod
  contains
  integer function my_func()
    my_func = 10
  end function my_func
end module my_mod

subroutine workshare(aa, bb, cc, dd, ee, ff, n)
  use my_mod
  integer n, i, j
  real aa(n), bb(n), cc(n), dd(n), ee(n), ff(n)

  !$omp workshare
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  aa = my_func()
  cc = dd
  ee = ff

  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  where (aa .ne. my_func()) aa = bb * cc
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  where (dd .lt. 5) dd = aa * my_func()

  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  where (aa .ge. my_func())
    !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
    cc = aa + my_func()
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  elsewhere (aa .le. my_func())
    !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
    cc = dd + my_func()
  elsewhere
    !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
    cc = ee + my_func()
  end where

  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  forall (j = 1:my_func()) aa(j) = aa(j) + bb(j)

  forall (j = 1:10)
    aa(j) = aa(j) + bb(j)

    !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
    where (cc .le. j) cc = cc + my_func()
  end forall

  !$omp atomic update
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  j = j + my_func()

  !$omp atomic capture
  i = j
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKSHARE construct
  j = j - my_func()
  !$omp end atomic

  !$omp end workshare

end subroutine workshare
