! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for duplicate definitions and initializations, mostly of procedures
module m
  procedure(real), pointer :: p
  !ERROR: EXTERNAL attribute was already specified on 'p'
  !ERROR: POINTER attribute was already specified on 'p'
  !ERROR: The type of 'p' has already been declared
  procedure(integer), pointer :: p
end

module m1
    real, dimension(:), pointer :: realArray => null()
    !ERROR: POINTER attribute was already specified on 'realarray'
    !ERROR: The type of 'realarray' has already been declared
    real, dimension(:), pointer :: realArray => localArray
end module m1

module m2
  interface
    subroutine sub()
    end subroutine sub
  end interface

  procedure(sub), pointer :: p1 => null()
  !ERROR: EXTERNAL attribute was already specified on 'p1'
  !ERROR: POINTER attribute was already specified on 'p1'
  !ERROR: The interface for procedure 'p1' has already been declared
  procedure(sub), pointer :: p1 => null()

end module m2

module m3
  interface
    real function fun()
    end function fun
  end interface

  procedure(fun), pointer :: f1 => null()
  !ERROR: EXTERNAL attribute was already specified on 'f1'
  !ERROR: POINTER attribute was already specified on 'f1'
  !ERROR: The interface for procedure 'f1' has already been declared
  procedure(fun), pointer :: f1 => null()

end module m3

module m4
  real, dimension(:), pointer :: localArray => null()
  type :: t2
    real, dimension(:), pointer :: realArray => null()
    !ERROR: Component 'realarray' is already declared in this derived type
    real, dimension(:), pointer :: realArray => localArray
  end type
end module m4

module m5
  !ERROR: Actual argument for 'string=' has bad type 'REAL(4)'
  character(len=len(a)) :: b
  !ERROR: The type of 'a' has already been implicitly declared
  character(len=len(b)) :: a
end module m5

module m6
  integer, dimension(3) :: iarray
  !ERROR: Derived type 'ubound' not found
  character(len=ubound(iarray)(1)) :: first
end module m6

module m7
  integer, dimension(2) :: iarray
  !ERROR: Derived type 'ubound' not found
  integer :: ivar = ubound(iarray)(1)
end module m7

module m8
  integer :: iVar = 3
  !ERROR: The type of 'ivar' has already been declared
  integer :: iVar = 4
  integer, target :: jVar = 5
  integer, target :: kVar = 5
  integer, pointer :: pVar => jVar
  !ERROR: POINTER attribute was already specified on 'pvar'
  !ERROR: The type of 'pvar' has already been declared
  integer, pointer :: pVar => kVar
end module m8

module m9
  integer :: p, q
  procedure() p ! ok
  !ERROR: The type of 'q' has already been declared
  procedure(real) q
end module m9
