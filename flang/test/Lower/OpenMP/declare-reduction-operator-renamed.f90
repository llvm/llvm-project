! Cross-module user-defined operator declare reduction, renamed on import
! (USE m, ONLY: operator(.local.) => operator(.remote.)). The declare-reduction
! op must be named from the source operator spelling (.remote.), never the
! use-site rename (.local.): lowering resolves the renamed clause operator back
! to the module's source reduction symbol. Reproducer prints 100 at run time
! (repro/B_renamed.f90). https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
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

program main
  use m, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program

! The op keys on the source spelling op.remote. (module-scoped); the use-site
! rename op.local. must never appear as an op name.
! loose capture (R1): the exact mangled qualifier is pinned after building.
! CHECK-NOT: op.local.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: op.local.
