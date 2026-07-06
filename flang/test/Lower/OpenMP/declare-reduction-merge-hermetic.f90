! A user-defined operator reduction re-exported through a HERMETIC facade (which
! embeds a private copy of the defining module) and reached both directly and
! through that facade must compile without a spurious ambiguity error: the direct
! and embedded copies are the same reduction. This guards the reduction ambiguity
! check, whose distinctness is by defining-module name and reduction name rather
! than by symbol pointer (a hermetic embed gives the same reduction two distinct
! ultimate symbols). The lowering of the merged reduction is covered by the
! multi-type user-defined reduction lowering support.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp hm_base.f90
! RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -fopenmp hm_facade.f90
! Reaching the reduction through both the base and the hermetic facade must not be
! diagnosed as ambiguous (the two copies are the same reduction).
! RUN: %flang_fc1 -fsyntax-only -fopenmp hm_use.f90

!--- hm_base.f90
module hm_base
  implicit none
  interface operator(.myop.)
    module procedure hm_f
  end interface
  !$omp declare reduction(.myop.:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
contains
  integer function hm_f(x, y)
    integer, intent(in) :: x, y
    hm_f = x * y
  end function
end module

!--- hm_facade.f90
module hm_facade
  use hm_base
end module

!--- hm_use.f90
program main
  use hm_base
  use hm_facade
  integer :: x, i
  x = 1
  !$omp parallel do reduction(.myop.:x)
  do i = 1, 5
    x = hm_f(x, i)
  end do
  print *, x
end program
