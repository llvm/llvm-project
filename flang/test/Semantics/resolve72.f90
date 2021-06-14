! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C709 An assumed-type entity shall be a dummy data object that does not have 
! the ALLOCATABLE, CODIMENSION, INTENT (OUT), POINTER, or VALUE attribute and 
! is not an explicit-shape array.
subroutine s()
  !ERROR: Assumed-type entity 'starvar' must be a dummy argument
  type(*) :: starVar

    contains
      subroutine inner1(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        type(*) :: arg1 ! OK        
        type(*), dimension(*) :: arg2 ! OK        
        !ERROR: Assumed-type argument 'arg3' cannot have the ALLOCATABLE attribute
        type(*), allocatable :: arg3
        !ERROR: Assumed-type argument 'arg4' cannot be a coarray
        type(*), codimension[*] :: arg4
        !ERROR: Assumed-type argument 'arg5' cannot be INTENT(OUT)
        type(*), intent(out) :: arg5
        !ERROR: Assumed-type argument 'arg6' cannot have the POINTER attribute
        type(*), pointer :: arg6
        !ERROR: Assumed-type argument 'arg7' cannot have the VALUE attribute
        type(*), value :: arg7
        !ERROR: Assumed-type array argument 'arg8' must be assumed shape, assumed size, or assumed rank
        type(*), dimension(3) :: arg8
      end subroutine inner1
end subroutine s
