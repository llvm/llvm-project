! RUN: %python %S/test_errors.py %s %flang_fc1
! Test restrictions on what subprograms can be used for defined operators.
! See: 15.4.3.4.2

module m1
  interface operator(+)
    !ERROR: OPERATOR(+) procedure 'add1' must be a function
    subroutine add1(x, y, z)
      real, intent(out) :: x
      real, intent(in) :: y, z
    end
  end interface
end

module m2
  interface operator(-)
    real function sub1(x)
      logical, intent(in) :: x
    end
    real function sub2(x, y)
      logical, intent(in) :: x, y
    end
    !ERROR: OPERATOR(-) function 'sub3' must have one or two dummy arguments
    real function sub3(x, y, z)
      real, intent(in) :: x, y, z
    end
  end interface
  interface operator(.not.)
    !ERROR: OPERATOR(.NOT.) function 'not1' must have one dummy argument
    real function not1(x, y)
      real, intent(in) :: x, y
    end
  end interface
end

module m3
  interface operator(/)
    !ERROR: A function interface may not declare an assumed-length CHARACTER(*) result
    character(*) function divide(x, y)
      character(*), intent(in) :: x, y
    end
  end interface
  interface operator(<)
    !ERROR: In OPERATOR(<) function 'lt1', dummy argument 'x' must have INTENT(IN) or VALUE attribute
    !ERROR: In OPERATOR(<) function 'lt1', dummy argument 'y' may not be OPTIONAL
    logical function lt1(x, y)
      logical :: x
      real, value, optional :: y
    end
    !ERROR: In OPERATOR(<) function 'lt2', dummy argument 'y' must be a data object
    logical function lt2(x, y)
      logical, intent(in) :: x
      intent(in) :: y
      interface
        subroutine y()
        end
      end interface
    end
  end interface
 contains
  subroutine s(alcf1, alcf2)
    interface
      character(*) function alcf1(x, y)
        character(*), intent(in) :: x, y
      end function
      character(*) function alcf2(x, y)
        character(*), intent(in) :: x, y
      end function
    end interface
    interface operator(+)
      !ERROR: OPERATOR(+) function 'alcf1' may not have assumed-length CHARACTER(*) result
      procedure alcf1
    end interface
    !ERROR: OPERATOR(-) function 'alcf2' may not have assumed-length CHARACTER(*) result
    generic :: operator(-) => alcf2
  end subroutine
end

module m4
  interface operator(+)
    !ERROR: OPERATOR(+) function 'add' conflicts with intrinsic operator
    complex function add(x, y)
      real, intent(in) :: x
      integer, value :: y
    end
    !ERROR: OPERATOR(+) function 'plus' conflicts with intrinsic operator
    real function plus(x)
      complex, intent(in) :: x
    end
  end interface
  interface operator(.not.)
    !WARNING: The external interface 'not1' is not compatible with an earlier definition (distinct numbers of dummy arguments)
    real function not1(x)
      real, value :: x
    end
    !ERROR: OPERATOR(.NOT.) function 'not2' conflicts with intrinsic operator
    logical(8) function not2(x)
      logical(8), value :: x
    end
  end interface
  interface operator(.and.)
    !ERROR: OPERATOR(.AND.) function 'and' conflicts with intrinsic operator
    real function and(x, y)
      logical(1), value :: x
      logical(8), value :: y
    end
  end interface
  interface operator(//)
    real function concat1(x, y)
      real, value :: x, y
    end
    real function concat2(x, y)
      character(kind=1, len=4), intent(in) :: x
      character(kind=4, len=4), intent(in) :: y
    end
    !ERROR: OPERATOR(//) function 'concat3' conflicts with intrinsic operator
    real function concat3(x, y)
      character(kind=4, len=4), intent(in) :: x
      character(kind=4, len=4), intent(in) :: y
    end
  end interface
end
