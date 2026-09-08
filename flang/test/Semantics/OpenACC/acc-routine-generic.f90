! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenacc

! A pure generic interface name is not a valid target of an OpenACC ROUTINE
! directive. The name argument denotes a subroutine or function; the same
! restriction appears in OpenMP for declare target procedure list items.
!
! Exception: when a generic shares its name with a specific procedure (a
! common pattern with interface bodies), the name denotes that specific and
! the directive is accepted.

module m1
  interface foo
    module procedure sbar, dbar
  end interface
  !ERROR: A generic interface name may not appear in an ACC ROUTINE clause: foo
  !$acc routine(foo) seq
contains
  real function sbar(x)
    real, intent(in) :: x
    sbar = x
  end function
  double precision function dbar(x)
    double precision, intent(in) :: x
    dbar = x
  end function
end module

! Same-named specific inside the interface: routine attaches to the specific.
module m2
  interface generic_sub
    subroutine generic_sub() bind(c)
    end subroutine
    subroutine other_sub(value) bind(c)
      integer(8), value :: value
    end subroutine
  end interface
  !$acc routine(generic_sub) seq bind("device_generic_sub")
end module

program gen
  use m1
  real :: y = 1.0, z
  !ERROR: A generic interface name may not appear in an ACC ROUTINE clause: foo
  !$acc routine(foo) seq
  !$acc parallel
  z = foo(y)
  !$acc end parallel
  print *, z
end program
