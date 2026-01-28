! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60
! OpenMP Version 6.0
! workdistribute Construct
! The !omp workdistribute construct must not contain any user defined
! function calls unless the function is ELEMENTAL.

module my_mod
  contains
  integer function my_func()
    my_func = 10
  end function my_func

  impure integer function impure_my_func()
    impure_my_func = 20
  end function impure_my_func

  impure elemental integer function impure_ele_my_func()
    impure_ele_my_func = 20
  end function impure_ele_my_func
end module my_mod

subroutine workdistribute(aa, bb, cc, n)
  use my_mod
  integer n
  real aa(n), bb(n), cc(n)
  !$omp teams
  !$omp workdistribute
  !ERROR: User defined non-ELEMENTAL function 'my_func' is not allowed in a WORKDISTRIBUTE construct
  aa = my_func()
  aa = bb * cc
  !$omp end workdistribute
  !$omp end teams

end subroutine workdistribute
