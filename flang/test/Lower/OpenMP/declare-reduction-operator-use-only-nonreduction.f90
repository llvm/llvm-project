! An operator declare reduction imported only incidentally: the consumer uses
! the defining module for an unrelated public entity (`use red_mod, only: token`)
! and never names the reduction's operator. The materializer walks every loaded
! mod-file module scope, so without an accessibility guard it would materialize
! this module's reduction op even though the program does not import it. Such an
! op is dead, and its combiner may reference a module-private helper, so it must
! not be emitted. A program that actually named the operator in a clause still
! gets its op (see declare-reduction-operator-separate.f90).
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp red_mod.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 use_only.f90 -o - | FileCheck use_only.f90

!--- red_mod.f90
module red_mod
  integer, parameter :: token = 42
  type :: t
    integer :: val = 0
  end type
  interface operator(.remote.)
    module procedure add_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

!--- use_only.f90
! The reduction op for red_mod is not materialized: the program imports only the
! unrelated `token`, never the operator, so the reduction is inaccessible. The
! program still lowers normally.
! CHECK-NOT: omp.declare_reduction @{{.*}}op.remote.
! CHECK-NOT: not yet implemented
! CHECK: func.func @_QQmain
program main
  use red_mod, only: token
  integer :: s
  s = token
  print *, s
end program
