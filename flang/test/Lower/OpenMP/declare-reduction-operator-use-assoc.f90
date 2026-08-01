! Cross-module USE-associated user-defined operator declare reduction (plain USE).
! The reduction clause names an operator (.plus.) whose declare reduction is
! imported from a module; lowering resolves it to the module's source reduction
! op and binds the clause to it. This used to be a clean TODO and was asserted by
! Todo/declare-reduction-operator-use-assoc.f90 (now removed / flipped to this
! positive test). Reproducer prints 100 at run time (repro/A_use_assoc.f90).
! https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t
    integer :: val = 0
  end type
  interface operator(.plus.)
    module procedure add_t
  end interface
  !$omp declare reduction(.plus.:t:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t(0))
contains
  type(t) function add_t(a, b)
    type(t), intent(in) :: a, b
    add_t%val = a%val + b%val
  end function
end module

program main
  use m
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.plus.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

! The op is named from the source module reduction symbol (module-scoped mangled
! "_QQ...op.plus.", not the bare use-site spelling), and the wsloop clause binds
! that same op.
! loose captures (R1): the exact mangled qualifier is pinned after building.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.plus\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK-NOT: omp.declare_reduction @op.plus.
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
