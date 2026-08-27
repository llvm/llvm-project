!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

! OpenMP 6.1 introduced modifier groups, and the ADJUST_ARGS clause now accepts
! the 'adjust-op' modifier group. The group is required, meaning that at least
! one modifier from the group must appear on the clause.
subroutine f00
!ERROR: ADJUST_ARGS clause on the DECLARE VARIANT directive is not yet implemented
!ERROR: modifier from 'adjust-op' modifier group is required
  !$omp declare variant (sub:vsub) match (construct={dispatch}) adjust_args(obj)
contains
  subroutine sub(x)
    integer :: x
  end
  subroutine vsub(x, obj)
    integer :: x, obj
  end
end

subroutine f01
!ERROR: ADJUST_ARGS clause on the DECLARE VARIANT directive is not yet implemented
!ERROR: modifier from 'adjust-op' modifier group is required
  !$omp declare variant (sub:vsub) match (construct={dispatch}) adjust_args(nothing, need_device_ptr: obj)
contains
  subroutine sub(x)
    integer :: x
  end
  subroutine vsub(x, obj)
    integer :: x, obj
  end
end
