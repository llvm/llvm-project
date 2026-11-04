!RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -Wunused-variable -Wused-undefined-variable
subroutine testUnusedWarnings(dummyArgument)
  common inCommon ! ok
  !WARNING: Value of local variable 'uninitializedunused' is never used [-Wunused-variable]
  real uninitializedUnused
  !WARNING: Value of uninitialized local variable 'undefined' is used but never defined [-Wused-undefined-variable]
  real undefined
  integer :: initialized = 1 ! ok
  !WARNING: Value of local variable 'unusedinitialized' is never used [-Wunused-variable]
  integer :: unusedInitialized = 2
  !WARNING: Value of local variable 'craypointer1' is never used [-Wunused-variable]
  !WARNING: Value of local variable 'craypointee1' is never used [-Wunused-variable]
  pointer (crayPointer1, crayPointee1)
  !WARNING: Value of uninitialized local variable 'craypointer2' is used but never defined [-Wused-undefined-variable]
  !WARNING: Value of uninitialized local variable 'craypointee2' is used but never defined [-Wused-undefined-variable]
  pointer (crayPointer2, crayPointee2)
  integer, parameter :: namedConstant = 123 ! ok
  real, target :: target ! ok
  real, pointer :: pointer ! ok
  equivalence (eq1, eq2) ! ok
  !WARNING: Value of uninitialized local variable 'a1' is used but never defined [-Wused-undefined-variable]
  real, allocatable :: a1, a2, a3, a4, a5
  !WARNING: Value of local variable 'b2' is never used [-Wunused-variable]
  real b1, b2
  allocate(a2, source=a1)
  allocate(a3, source=a2)
  do j = 1, 10 ! ok
  end do
  pointer => target
  call move_alloc(a4, a5)
  read(*,*) b1
  print *, undefined, initialized, crayPointee2, a3, b1
end
