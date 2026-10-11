! Separate-compilation same-mangled reduction collision, one imported and one not:
! two modules (a_mod, b_mod) each declare the identical mangled reduction
! (operator(.plus.) over the same type t from a shared base module). The consumer
! imports a_mod's operator fully (use a_mod, only: operator(.plus.)) and reduces
! with it, but imports only a non-reduction token from b_mod
! (use b_mod, only: token). Only a_mod's reduction is actually reachable, so only
! it may be materialized; b_mod's is a dead op the program never imports.
!
! Resolving a candidate's mangled name (op.plus.) from the program-unit scope
! finds a_mod's reduction. Because both modules share that mangled name, marking
! must confirm the resolved reduction IS this candidate (pointer identity on the
! ultimate) before accepting it, so b_mod's unimported reduction is refused.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp base.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp a_mod.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp b_mod.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 prog.f90 -o - \
! RUN:   | FileCheck --implicit-check-not='Mb_modop.plus.' prog.f90

!--- base.f90
module base
  type :: t
    integer :: v = 0
  end type
end module

!--- a_mod.f90
module a_mod
  use base, only: t
  interface operator(.plus.)
    module procedure add_a
  end interface
  !$omp declare reduction(.plus.:t:omp_out%v=omp_out%v+omp_in%v) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_a(x, y)
    type(t), intent(in) :: x, y
    add_a%v = x%v + y%v
  end function
end module

!--- b_mod.f90
module b_mod
  use base, only: t
  integer, parameter :: token = 42
  interface operator(.plus.)
    module procedure add_b
  end interface
  !$omp declare reduction(.plus.:t:omp_out%v=omp_out%v+omp_in%v) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_b(x, y)
    type(t), intent(in) :: x, y
    add_b%v = x%v + y%v
  end function
end module

!--- prog.f90
! a_mod's reduction is materialized and bound; b_mod's same-mangled reduction is
! NOT materialized (the --implicit-check-not above rejects its module-qualified op
! name anywhere in the output), and no TODO aborts the compile.
! CHECK: omp.declare_reduction @{{.*}}a_mod{{.*}}op.plus.
! CHECK-NOT: not yet implemented
program main
  use base, only: t
  use a_mod, only: operator(.plus.)
  use b_mod, only: token
  type(t) :: x
  integer :: k
  x = t(0)
  !$omp parallel do reduction(.plus.:x)
  do k = 1, 10
    x = x .plus. t(k)
  end do
  print *, x%v, token
end program
