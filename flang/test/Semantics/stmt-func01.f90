! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! C1577
program main
  type t1(k,l)
    integer, kind :: k = kind(1)
    integer, len :: l = 666
    integer(k) n
  end type t1
  interface
    pure integer function ifunc()
    end function
  end interface
  !PORTABILITY: Automatic data object 'x1' should not appear in the specification part of a main program
  type(t1(k=4,l=ifunc())) x1
  !PORTABILITY: Statement function 'sf1' should not contain an array constructor
  sf1(n) = sum([(j,j=1,n)])
  type(t1) sf2
  !PORTABILITY: Statement function 'sf2' should not contain a structure constructor
  sf2(n) = t1(n)
  !PORTABILITY: Statement function 'sf3' should not contain a type parameter inquiry
  sf3(n) = x1%l
  !ERROR: Recursive call to statement function 'sf4' is not allowed
  sf4(n) = sf4(n)
  !ERROR: Statement function 'sf5' may not reference another statement function 'sf6' that is defined later
  sf5(n) = sf6(n)
  real sf7
  !ERROR: Statement function 'sf6' may not reference another statement function 'sf7' that is defined later
  sf6(n) = sf7(n)
  !PORTABILITY: Statement function 'sf7' should not reference function 'explicit' that requires an explicit interface
  sf7(n) = explicit(n)
  real :: a(3) = [1., 2., 3.]
  !PORTABILITY: Statement function 'sf8' should not pass an array argument that is not a whole array
  sf8(n) = sum(a(1:2))
  sf8a(n) = sum(a) ! ok
  integer :: sf9
  !ERROR: Defining expression of statement function 'sf9' cannot be converted to its result type INTEGER(4)
  sf9(n) = "bad"
  !ERROR: Statement function 'sf10' may not reference another statement function 'sf11' that is defined later
  sf10(n) = sf11(n)
  sf11(n) = sf10(n) ! mutual recursion, caused crash
  integer(1) iarg1
  !PORTABILITY: nonstandard usage: based POINTER
  pointer(iarg1p, iarg1)
  sf13(iarg1) = iarg1
  ! executable part
  print *, sf13(iarg1) ! ok
  sf14 = 1.
 contains
  real function explicit(x,y)
    integer, intent(in) :: x
    integer, intent(in), optional :: y
    explicit = x
  end function
  pure function arr()
    real :: arr(2)
    arr = [1., 2.]
  end function
  subroutine foo
    !PORTABILITY: An implicitly typed statement function should not appear when the same symbol is available in its host scope
    sf14(x) = 2.*x
  end subroutine
end

subroutine s0
  allocatable :: sf
  !ERROR: 'sf' is not a callable procedure
  sf(x) = 1.
end

subroutine s1
  asynchronous :: sf
  !ERROR: An entity may not have the ASYNCHRONOUS attribute unless it is a variable
  sf(x) = 1.
end

subroutine s2
  pointer :: sf
  !ERROR: A statement function must not have the POINTER attribute
  sf(x) = 1.
end

subroutine s3
  save :: sf
  !ERROR: The entity 'sf' with an explicit SAVE attribute must be a variable, procedure pointer, or COMMON block
  sf(x) = 1.
end

subroutine s4
  volatile :: sf
  !ERROR: VOLATILE attribute may apply only to a variable
  sf(x) = 1.
end

subroutine s5
  !ERROR: Invalid specification expression: reference to impure function 'k'
  real x(k())
  !WARNING: Name 'k' from host scope should have a type declaration before its local statement function definition
  !ERROR: 'k' is already declared in this scoping unit
  k() = 0.0
end
