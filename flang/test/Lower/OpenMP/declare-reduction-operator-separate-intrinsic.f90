! Separate-compilation intrinsic-operator user declare reduction: a module
! declares reduction(+:dt:...) over a derived type through an intrinsic
! operator(+) interface, compiled on its own to a .mod. A consumer imports the
! type and the operator (use m, only: dt, operator(+)) and reduces with the
! intrinsic operator. Lowering the consumer must materialize the imported
! reduction so the clause binds, rather than fall through to a "not yet
! implemented" TODO.
!
! Accessibility must recognize the intrinsic operator: the reduction is not
! visible under its own mangled name (op.+), only through the imported
! operator(+). Resolving with the shared clause-side resolver, which maps op.+ to
! operator(+) and traces it to the source reduction, handles this.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp m.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 use.f90 -o - | FileCheck use.f90

!--- m.f90
module m
  type :: dt
    integer :: i = 0
  end type
  interface operator(+)
    module procedure add_dt
  end interface
  !$omp declare reduction(+:dt:omp_out=omp_out+omp_in) initializer(omp_priv=dt(0))
contains
  function add_dt(a, b) result(r)
    type(dt), intent(in) :: a, b
    type(dt) :: r
    r%i = a%i + b%i
  end function
end module

!--- use.f90
! The intrinsic-operator reduction is materialized under its module-scoped name
! (getScopedUserReductionName, byte-identical on the clause side) and the clause
! binds it, so no TODO aborts the compile.
! CHECK: omp.declare_reduction @"[[RED:_QQ[A-Za-z0-9_.]*op\.\+[A-Za-z0-9_.]*]]" : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @"[[RED]]"
! CHECK-NOT: not yet implemented
program main
  use m, only: dt, operator(+)
  type(dt) :: x
  integer :: k
  x = dt(0)
  !$omp parallel do reduction(+:x)
  do k = 1, 10
    x = x + dt(k)
  end do
  print *, x%i
end program
