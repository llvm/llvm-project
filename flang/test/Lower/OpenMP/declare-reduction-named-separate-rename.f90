! Separate-compilation NAMED user declare reduction imported under a rename: a
! module declares a named reduction (myred) over a derived type, compiled on its
! own to a .mod. The consumer imports it under an alias
! (use m_named, only: dt, alias => myred) and reduces with the alias. Lowering
! the consumer must materialize the imported reduction so the clause binds,
! rather than fall through to a "not yet implemented" TODO.
!
! Accessibility cannot key on the reduction's source name here: the program sees
! only `alias`, so a name-driven lookup for `myred` misses it. The alias' ultimate
! symbol IS the reduction, so scanning the program-unit scope for a visible symbol
! whose ultimate carries UserReductionDetails recovers it regardless of spelling.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp m_named.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 use.f90 -o - | FileCheck use.f90

!--- m_named.f90
module m_named
  type :: dt
    integer :: i = 0
  end type
  !$omp declare reduction(myred:dt:omp_out%i=omp_out%i+omp_in%i) &
  !$omp   initializer(omp_priv=dt(0))
end module

!--- use.f90
! The named reduction, imported under the alias, is still materialized (named from
! its source module) and the clause binds it, so no TODO aborts the compile.
! CHECK: omp.declare_reduction @{{.*}}myred{{.*}} : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @{{.*}}myred
! CHECK-NOT: not yet implemented
program main
  use m_named, only: dt, alias => myred
  type(dt) :: x
  integer :: k
  x = dt(0)
  !$omp parallel do reduction(alias:x)
  do k = 1, 10
    x%i = x%i + k
  end do
  print *, x%i
end program
