! Cross-module user-defined operator declare reduction, a regression against a
! silent miscompile. Two modules (mod_add, mod_mul) declare operator(.remote.)
! and a declare reduction for the same derived type t (from tmod), but with
! different combiners (sum vs product) and matching identity initializers
! (0 vs 1). The program imports both, each renamed to a distinct use-site
! spelling (.addop., .mulop.), and reduces one variable with each.
!
! The reduction op name keys on the resolved source symbol's (name, owner), so
! the two same-spelling same-type reductions get distinct module-scoped op names
! and each clause binds its own combiner. Binding both clauses to one op would
! run one variable through the other's combiner. The test asserts the two ops
! are distinct (different owning-module qualifier) and that each clause binds its
! own; a collision would emit a single op or cross-bind, which FileCheck catches.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module tmod
  type :: t
    integer :: val = 0
  end type
end module

module mod_add
  use tmod
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

module mod_mul
  use tmod
  interface operator(.remote.)
    module procedure mul_t
  end interface
  !$omp declare reduction(.remote.:t:omp_out%val=omp_out%val*omp_in%val) &
  !$omp   initializer(omp_priv=t(1))
contains
  type(t) function mul_t(a, b)
    type(t), intent(in) :: a, b
    mul_t%val = a%val * b%val
  end function
end module

program main
  use tmod, only: t
  use mod_add, only: operator(.addop.) => operator(.remote.)
  use mod_mul, only: operator(.mulop.) => operator(.remote.)
  type(t) :: x, y
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.addop.:x)
  do i = 1, 5
    x%val = x%val + i
  end do
  y = t(1)
  !$omp parallel do reduction(.mulop.:y)
  do i = 1, 5
    y%val = y%val * i
  end do
  print '(I0,1X,I0)', x%val, y%val
end program

! Two distinct module-scoped ops (mod_add vs mod_mul owner qualifier), each bound
! by its own clause (the addop loop is emitted first, then the mulop loop).
! loose captures (R1) keyed on the owning module name; the two qualifiers differ.
! CHECK-DAG: omp.declare_reduction @[[REDADD:_QQ[A-Za-z0-9_.]*mod_add[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK-DAG: omp.declare_reduction @[[REDMUL:_QQ[A-Za-z0-9_.]*mod_mul[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[REDADD]]
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[REDMUL]]
