! RUN: %S/test_errors.sh %s %flang %t
! Test 15.5.2.9(2,3,5) dummy procedure requirements

module m
 contains

  integer function intfunc(x)
    integer, intent(in) :: x
    intfunc = x
  end function
  real function realfunc(x)
    real, intent(in) :: x
    realfunc = x
  end function

  subroutine s01(p)
    procedure(realfunc), pointer, intent(in) :: p
  end subroutine
  subroutine s02(p)
    procedure(realfunc), pointer :: p
  end subroutine

  subroutine selemental1(p)
    procedure(cos) :: p ! ok
  end subroutine

  real elemental function elemfunc(x)
    real, intent(in) :: x
    elemfunc = x
  end function
  !ERROR: A dummy procedure may not be ELEMENTAL
  subroutine selemental2(p)
    procedure(elemfunc) :: p
  end subroutine

  function procptr()
    procedure(realfunc), pointer :: procptr
    procptr => realfunc
  end function
  function intprocptr()
    procedure(intfunc), pointer :: intprocptr
    intprocptr => intfunc
  end function

  subroutine test1 ! 15.5.2.9(5)
    procedure(realfunc), pointer :: p
    procedure(intfunc), pointer :: ip
    p => realfunc
    ip => intfunc
    call s01(realfunc) ! ok
    !ERROR: Actual argument procedure has interface incompatible with dummy argument 'p='
    call s01(intfunc)
    call s01(p) ! ok
    call s01(procptr()) ! ok
    !ERROR: Actual argument procedure has interface incompatible with dummy argument 'p='
    call s01(intprocptr())
    call s01(null()) ! ok
    call s01(null(p)) ! ok
    !ERROR: Actual argument procedure has interface incompatible with dummy argument 'p='
    call s01(null(ip))
    call s01(sin) ! ok
    !ERROR: Actual argument associated with procedure pointer dummy argument 'p=' must be a POINTER unless INTENT(IN)
    call s02(realfunc)
    call s02(p) ! ok
    !ERROR: Actual argument procedure has interface incompatible with dummy argument 'p='
    call s02(ip)
    !ERROR: Actual argument associated with procedure pointer dummy argument 'p=' must be a POINTER unless INTENT(IN)
    call s02(procptr())
    !ERROR: Actual argument associated with procedure pointer dummy argument 'p=' must be a POINTER unless INTENT(IN)
    call s02(null())
    !ERROR: Actual argument associated with procedure pointer dummy argument 'p=' must be a POINTER unless INTENT(IN)
    call s02(null(p))
    !ERROR: Actual argument associated with procedure pointer dummy argument 'p=' must be a POINTER unless INTENT(IN)
    call s02(sin)
  end subroutine

  subroutine callsub(s)
    call s
  end subroutine
  subroutine takesrealfunc1(f)
    external f
    real f
  end subroutine
  subroutine takesrealfunc2(f)
    x = f(1)
  end subroutine
  subroutine forwardproc(p)
    implicit none
    external :: p ! function or subroutine not known
    call foo(p)
  end subroutine

  subroutine test2(unknown,ds,drf,dif) ! 15.5.2.9(2,3)
    external :: unknown, ds, drf, dif
    real :: drf
    integer :: dif
    procedure(callsub), pointer :: ps
    procedure(realfunc), pointer :: prf
    procedure(intfunc), pointer :: pif
    call ds ! now we know that's it's a subroutine
    call callsub(callsub) ! ok apart from infinite recursion
    call callsub(unknown) ! ok
    call callsub(ds) ! ok
    call callsub(ps) ! ok
    call takesrealfunc1(realfunc) ! ok
    call takesrealfunc1(unknown) ! ok
    call takesrealfunc1(drf) ! ok
    call takesrealfunc1(prf) ! ok
    call takesrealfunc2(realfunc) ! ok
    call takesrealfunc2(unknown) ! ok
    call takesrealfunc2(drf) ! ok
    call takesrealfunc2(prf) ! ok
    call forwardproc(callsub) ! ok
    call forwardproc(realfunc) ! ok
    call forwardproc(intfunc) ! ok
    call forwardproc(unknown) ! ok
    call forwardproc(ds) ! ok
    call forwardproc(drf) ! ok
    call forwardproc(dif) ! ok
    call forwardproc(ps) ! ok
    call forwardproc(prf) ! ok
    call forwardproc(pif) ! ok
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(realfunc)
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(intfunc)
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(drf)
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(dif)
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(prf)
    !ERROR: Actual argument associated with procedure dummy argument 's=' is a function but must be a subroutine
    call callsub(pif)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc1(callsub)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc1(ds)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc1(ps)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc1(intfunc)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc1(dif)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc1(pif)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc1(intfunc)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc2(callsub)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc2(ds)
    !ERROR: Actual argument associated with procedure dummy argument 'f=' is a subroutine but must be a function
    call takesrealfunc2(ps)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc2(intfunc)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc2(dif)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc2(pif)
    !ERROR: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
    call takesrealfunc2(intfunc)
  end subroutine
end module
