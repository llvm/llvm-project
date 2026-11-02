!RUN: %python %S/test_errors.py %s %flang_fc1

! UsageWarning::BindCArrayDescriptor (see bind-c20.f90) is off by default:
! an assumed-shape or assumed-rank BIND(C) dummy argument, which would warn
! under -pedantic, produces no diagnostic here without it.

subroutine assumedShape(a)
  interface
    subroutine cFunc(a) bind(c)
      real, intent(in) :: a(:)
    end subroutine
  end interface
  real :: a(10)
  call cFunc(a)
end subroutine

subroutine assumedRank(a)
  interface
    subroutine cFunc2(a) bind(c)
      real, intent(in) :: a(..)
    end subroutine
  end interface
  real :: a(10)
  call cFunc2(a)
end subroutine
