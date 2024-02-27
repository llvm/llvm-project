! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! !DIR$ IGNORE_TKR tests

!ERROR: !DIR$ IGNORE_TKR directive must appear in a subroutine or function
!dir$ ignore_tkr

module m

!ERROR: !DIR$ IGNORE_TKR directive must appear in a subroutine or function
!dir$ ignore_tkr

  interface
    subroutine t1(x)
!dir$ ignore_tkr
      real, intent(in) :: x
    end

    subroutine t2(x)
!dir$ ignore_tkr(t) x
      real, intent(in) :: x
    end

    subroutine t3(x)
!dir$ ignore_tkr(k) x
      real, intent(in) :: x
    end

    subroutine t4(a)
!dir$ ignore_tkr(r) a
      real, intent(in) :: a(2)
    end

    subroutine t5(m)
!dir$ ignore_tkr(r) m
      real, intent(in) :: m(2,2)
    end

    subroutine t6(x)
!dir$ ignore_tkr(a) x
      real, intent(in) :: x
    end

    subroutine t7(x)
!ERROR: !DIR$ IGNORE_TKR directive may not have an empty parenthesized list of letters
!dir$ ignore_tkr() x
      real, intent(in) :: x
    end

    subroutine t8(x)
!dir$ ignore_tkr x
      real, intent(in) :: x
    end

    subroutine t9(x)
!dir$ ignore_tkr x
!WARNING: !DIR$ IGNORE_TKR should not apply to an allocatable or pointer
      real, intent(in), allocatable :: x
    end

    subroutine t10(x)
!dir$ ignore_tkr x
!WARNING: !DIR$ IGNORE_TKR should not apply to an allocatable or pointer
      real, intent(in), pointer :: x
    end

    subroutine t11
!dir$ ignore_tkr x
!ERROR: !DIR$ IGNORE_TKR directive may apply only to a dummy data argument
      real :: x
    end

    subroutine t12(p,q,r)
!dir$ ignore_tkr p, q
!ERROR: 'p' is a data object and may not be EXTERNAL
      real, external :: p
!ERROR: 'q' is already declared as an object
      procedure(real) :: q
      procedure(), pointer :: r
!ERROR: 'r' must be an object
!dir$ ignore_tkr r
    end

    elemental subroutine t13(x)
!dir$ ignore_tkr(r) x
!ERROR: !DIR$ IGNORE_TKR(R) may not apply in an ELEMENTAL procedure
      real, intent(in) :: x
    end

    subroutine t14(x)
!dir$ ignore_tkr(r) x
!WARNING: !DIR$ IGNORE_TKR(R) should not apply to a dummy argument passed via descriptor
      real x(:)
    end

  end interface

 contains
    subroutine t15(x)
!dir$ ignore_tkr x
!ERROR: !DIR$ IGNORE_TKR may not apply to an allocatable or pointer
      real, intent(in), allocatable :: x
    end

    subroutine t16(x)
!dir$ ignore_tkr x
!ERROR: !DIR$ IGNORE_TKR may not apply to an allocatable or pointer
      real, intent(in), pointer :: x
    end

  subroutine t17(x)
    real x
    x = x + 1.
!ERROR: !DIR$ IGNORE_TKR directive must appear in the specification part
!dir$ ignore_tkr x
  end

  subroutine t18(x)
!ERROR: 'q' is not a valid letter for !DIR$ IGNORE_TKR directive
!dir$ ignore_tkr(q) x
    real x
    x = x + 1.
  end

  subroutine t19(x)
    real x
   contains
    subroutine inner
!ERROR: 'x' must be local to this subprogram
!dir$ ignore_tkr x
    end
  end

  subroutine t20(x)
    real x
    block
!ERROR: 'x' must be local to this subprogram
!dir$ ignore_tkr x
    end block
  end

  subroutine t22(x)
!dir$ ignore_tkr(r) x
!WARNING: !DIR$ IGNORE_TKR(R) is not meaningful for an assumed-rank array
    real x(..)
  end

  subroutine t23(x)
!dir$ ignore_tkr(r) x
!ERROR: !DIR$ IGNORE_TKR(R) may not apply to a dummy argument passed via descriptor
    real x(:)
  end

end

subroutine bad1(x)
!dir$ ignore_tkr x
!ERROR: !DIR$ IGNORE_TKR may apply only in an interface or a module procedure
  real, intent(in) :: x
end

program test

!ERROR: !DIR$ IGNORE_TKR directive must appear in a subroutine or function
!dir$ ignore_tkr

  use m
  real x
  real a(2)
  real m(2,2)
  double precision dx

  call t1(1)
  call t1(dx)
  call t1('a')
  call t1((1.,2.))
  call t1(.true.)

  call t2(1)
  !ERROR: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'REAL(4)'
  call t2(dx)
  call t2('a')
  call t2((1.,2.))
  call t2(.true.)

  !ERROR: Actual argument type 'INTEGER(4)' is not compatible with dummy argument type 'REAL(4)'
  call t3(1)
  call t3(dx)
  !ERROR: passing Hollerith or character literal as if it were BOZ
  call t3('a')
  !ERROR: Actual argument type 'COMPLEX(4)' is not compatible with dummy argument type 'REAL(4)'
  call t3((1.,2.))
  !ERROR: Actual argument type 'LOGICAL(4)' is not compatible with dummy argument type 'REAL(4)'
  call t3(.true.)

  call t4(x)
  call t4(m)
  call t5(x)
  !WARNING: Actual argument array has fewer elements (2) than dummy argument 'm=' array (4)
  call t5(a)

  call t6(1)
  call t6(dx)
  call t6('a')
  call t6((1.,2.))
  call t6(.true.)
  call t6(a)

  call t8(1)
  call t8(dx)
  call t8('a')
  call t8((1.,2.))
  call t8(.true.)
  call t8(a)

 contains
  subroutine inner(x)
!dir$ ignore_tkr x
!ERROR: !DIR$ IGNORE_TKR may apply only in an interface or a module procedure
    real, intent(in) :: x
  end
end
