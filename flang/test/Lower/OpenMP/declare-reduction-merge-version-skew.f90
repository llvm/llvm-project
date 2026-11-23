! Known limitation: two hermetic module files that embed DIFFERENT versions of a
! same-named module (an inconsistent build) each carry a reduction with the same
! operator and type but a different combiner. The reduction ambiguity check keys
! a reduction on (defining-module name, reduction name), which is identical for
! the two versions, so the clause is not diagnosed as ambiguous and the reduction
! is resolved by USE order. This matches the pre-existing behavior: Flang does not
! reject a same-named module loaded at two different versions. A stronger,
! content-based key would diagnose it; this test is the regression anchor for that
! future change. https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fopenmp vs_base_v1.f90
! RUN: %flang_fc1 -fhermetic-module-files -fopenmp vs_facade_a.f90
! RUN: %flang_fc1 -fopenmp vs_base_v2.f90
! RUN: %flang_fc1 -fhermetic-module-files -fopenmp vs_facade_b.f90
! The inconsistent build currently compiles with no ambiguity diagnostic (the
! known limitation). If a future content-based key starts diagnosing it, update
! this test.
! RUN: %flang_fc1 -fsyntax-only -fopenmp vs_use.f90

!--- vs_base_v1.f90
module vs_base
  implicit none
  interface operator(.myop.)
    module procedure vs_f
  end interface
  !$omp declare reduction(.myop.:integer:omp_out=omp_out+omp_in) &
  !$omp   initializer(omp_priv=0)
contains
  integer function vs_f(x, y)
    integer, intent(in) :: x, y
    vs_f = x + y
  end function
end module

!--- vs_facade_a.f90
module vs_facade_a
  use vs_base
end module

!--- vs_base_v2.f90
module vs_base
  implicit none
  interface operator(.myop.)
    module procedure vs_f
  end interface
  !$omp declare reduction(.myop.:integer:omp_out=omp_out*omp_in) &
  !$omp   initializer(omp_priv=1)
contains
  integer function vs_f(x, y)
    integer, intent(in) :: x, y
    vs_f = x * y
  end function
end module

!--- vs_facade_b.f90
module vs_facade_b
  use vs_base
end module

!--- vs_use.f90
program main
  use vs_facade_a
  use vs_facade_b
  integer :: x, i
  x = 1
  !$omp parallel do reduction(.myop.:x)
  do i = 1, 5
    x = x + i
  end do
  print *, x
end program
