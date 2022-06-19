! RUN: %python %S/test_errors.py %s %flang_fc1
! C708 An entity declared with the CLASS keyword shall be a dummy argument 
! or have the ALLOCATABLE or POINTER attribute.
subroutine s()
  type :: parentType
  end type

  class(parentType), pointer :: pvar
  class(parentType), allocatable :: avar
  class(*), allocatable :: starAllocatableVar
  class(*), pointer :: starPointerVar
  !ERROR: CLASS entity 'barevar' must be a dummy argument or have ALLOCATABLE or POINTER attribute
  class(parentType) :: bareVar
  !ERROR: CLASS entity 'starvar' must be a dummy argument or have ALLOCATABLE or POINTER attribute
  class(*) :: starVar

    contains
      subroutine inner(arg1, arg2, arg3, arg4, arg5)
        class (parenttype) :: arg1, arg3
        type(parentType) :: arg2
        class (parenttype), pointer :: arg4
        class (parenttype), allocatable :: arg5
      end subroutine inner
end subroutine s
