! RUN: %python %S/test_errors.py %s %flang_fc1 -fcoarray
pure subroutine coarray_from(f, t)
  integer, allocatable, intent(inout) :: f(:)[:], t(:)[:]
  !ERROR: Procedure 'move_alloc' referenced in pure subprogram 'coarray_from' must be pure too
  !ERROR: An image control statement may not appear in a pure subprogram
  call move_alloc(f, t)
end subroutine

pure subroutine noncoarray_from(f, t)
  integer, allocatable, intent(inout) :: f(:)
  integer, allocatable, intent(out) :: t(:)
  call move_alloc(f, t)
end subroutine
