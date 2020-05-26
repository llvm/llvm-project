! RUN: %S/test_errors.sh %s %t %f18
! Tests for duplicate definitions and initializations, mostly of procedures
module m
  procedure(real), pointer :: p
  !ERROR: The interface for procedure 'p' has already been declared
  procedure(integer), pointer :: p
end

module m1
    real, dimension(:), pointer :: realArray => null()
    !ERROR: The type of 'realarray' has already been declared
    real, dimension(:), pointer :: realArray => localArray
end module m1

module m2
  interface
    subroutine sub()
    end subroutine sub
  end interface

  procedure(sub), pointer :: p1 => null()
  !ERROR: The interface for procedure 'p1' has already been declared
  procedure(sub), pointer :: p1 => null()

end module m2

module m3
  interface
    real function fun()
    end function fun
  end interface

  procedure(fun), pointer :: f1 => null()
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
