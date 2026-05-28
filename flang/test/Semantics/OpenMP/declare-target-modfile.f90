!RUN: %python %S/../test_modfile.py %s %flang_fc1 -fopenmp -fopenmp-version=60

module m
integer :: x
!$omp declare_target link(x) device_type(nohost)
interface
  real function g(v)
    real :: v(10)
    !$omp declare_target
  end
end interface
contains
subroutine f
  !$omp declare_target(f)
end
subroutine h
  integer, save :: a(10)
  !$omp declare_target enter(h, a)
  continue
end
end module

!Expect: m.mod
!module m
!integer(4)::x
!interface
!function g(v)
!real(4)::v(1_8:10_8)
!real(4)::g
!end
!end interface
!!$omp declare_target device_type(nohost) link(x)
!!$omp declare_target enter(g)
!!$omp declare_target enter(f)
!!$omp declare_target enter(h)
!contains
!subroutine f()
!end
!subroutine h()
!end
!end
