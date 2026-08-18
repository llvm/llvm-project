! An unused import of an operator declare reduction over a type lowering does not
! yet support must not abort the consumer. The materializer eagerly walks every
! imported module reduction; without a support prefilter it would call the
! lowering path for this reduction and hit getReductionType's "not yet
! implemented" TODO, aborting a program that merely USEs the module. The
! materializer's isSimpleReductionType prefilter skips it, so nothing is emitted
! and the program compiles. A program that actually named this reduction in a
! clause would still get the clean clause-side TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t

! Produce the .mod with -fsyntax-only: lowering the module's own reduction would
! hit the same-file TODO (expected for the unsupported type), but semantics and
! the .mod write succeed.
! RUN: %flang_fc1 -fsyntax-only -fopenmp unsup.mod.f90
! The consumer USEs the type but never the reduction; it must lower cleanly.
! RUN: %flang_fc1 -emit-hlfir -fopenmp unsup.use.f90 -o - 2>&1 | FileCheck unsup.use.f90

! An unused import of the OpenMP 6.0 combiner-in-clause form (the combiner is in
! a clause, not the reduction specifier) must also compile: the materializer
! reads the combiner from the specifier (computeReductionType and the impl both
! do), so it cannot lower this form and skips it instead of crashing on the empty
! specifier combiner.
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=60 clause.mod.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp clause.use.f90 -o - 2>&1 | FileCheck clause.use.f90

!--- unsup.mod.f90
module red_unsup
  type :: tarr
    integer :: a(4) = 0
  end type
  interface operator(.remote.)
    module procedure add_tarr
  end interface
  !$omp declare reduction(.remote.:tarr:omp_out%a=omp_out%a+omp_in%a) &
  !$omp   initializer(omp_priv=tarr())
contains
  type(tarr) function add_tarr(a, b)
    type(tarr), intent(in) :: a, b
    add_tarr%a = a%a + b%a
  end function
end module

!--- unsup.use.f90
! No reduction op is materialized for the unused unsupported import, and no TODO
! aborts the compile; the program lowers normally.
! CHECK-NOT: omp.declare_reduction
! CHECK-NOT: not yet implemented
! CHECK: func.func @_QQmain
program main
  use red_unsup, only: tarr
  type(tarr) :: x
  x%a = 0
  print *, sum(x%a)
end program

!--- clause.mod.f90
module red_clause
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t) combiner(omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- clause.use.f90
! No op is materialized for the unused combiner-in-clause import, and no TODO or
! crash aborts the compile.
! CHECK-NOT: omp.declare_reduction
! CHECK-NOT: not yet implemented
! CHECK: func.func @_QQmain
program main
  use red_clause, only: t
  type(t) :: x
  x%val = 0
  print *, x%val
end program
