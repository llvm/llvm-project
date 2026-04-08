! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
subroutine s1
  type :: d1
    real :: x
  end type
  type :: d2
     type(d1) :: x
  end type
  type(d1), target :: a(5)
  type(d2), target :: b(5)
  real, pointer, contiguous :: c(:)
  c => a%x ! okay, type has single component
  c => b%x%x ! okay, types have single components
end

subroutine s2
  type :: d1
    real :: x, y
  end type
  type(d1), target :: b(5)
  real, pointer, contiguous :: c(:)
  !ERROR: CONTIGUOUS pointer may not be associated with a discontiguous target
  c => b%x
  c => b(1:1)%x ! okay, one element
  !ERROR: CONTIGUOUS pointer may not be associated with a discontiguous target
  c => b(1:2)%x
end
