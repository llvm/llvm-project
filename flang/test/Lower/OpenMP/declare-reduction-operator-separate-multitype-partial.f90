! Separate-compilation multi-type operator declare reduction where one listed type
! is lowerable and a sibling is not: a single declaration lists tok (a scalar
! derived type, supported) and tbad (a derived type with an array component, which
! isSimpleReductionType rejects). A separate consumer imports only tok and reduces
! it. The lazy, clause-driven materializer checks lowerability per TYPE (not the
! whole-declaration gate), so it materializes the requested tok op even though the
! tbad sibling is unsupported, and the tbad op is never emitted. The clause must
! bind the tok op with no "not yet implemented" TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! Produce the .mod with -fsyntax-only: lowering the module's own declaration would
! hit the same-file TODO on the unsupported tbad type, but semantics and the .mod
! write succeed.
! RUN: %flang_fc1 -fsyntax-only -fopenmp m.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp use.f90 -o - | FileCheck use.f90

!--- m.f90
module m
  type :: tok
    integer :: v = 0
  end type
  type :: tbad
    integer :: v(4) = 0
  end type
  interface operator(.op.)
    module procedure aok, abad
  end interface
  !$omp declare reduction(.op.:tok,tbad:omp_out%v=omp_out%v+omp_in%v)
contains
  type(tok) function aok(a, b)
    type(tok), intent(in) :: a, b
    aok%v = a%v + b%v
  end function
  type(tbad) function abad(a, b)
    type(tbad), intent(in) :: a, b
    abad%v = a%v + b%v
  end function
end module

!--- use.f90
! Exactly one op is materialized (for tok). The CHECK-NOT between the tok op and
! the loop rejects a spurious second op (which is what materializing the
! unsupported tbad sibling would produce). No TODO aborts the compile.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.op\.[A-Za-z0-9_.]*tok]] : !fir.ref
! CHECK-NOT: omp.declare_reduction
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use m, only: tok, operator(.op.)
  type(tok) :: x
  integer :: i
  x = tok(0)
  !$omp parallel do reduction(.op.:x)
  do i = 1, 5
    x%v = x%v + i
  end do
  print *, x%v
end program
