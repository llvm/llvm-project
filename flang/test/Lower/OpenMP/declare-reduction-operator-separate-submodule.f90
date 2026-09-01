! Separate-compilation user-defined operator declare reduction referenced from a
! reduction clause inside a separately compiled submodule. The reduction lives in
! module rmod (compiled on its own to a .mod); an ancestor module pm2 declares a
! separate module procedure w; and the submodule smU implements w with a
! reduction(.plusb.:...) clause that names rmod's imported operator. Lowering the
! submodule is a distinct translation unit that never lowers rmod's own
! declaration, so the clause must lazily materialize the imported reduction op
! rather than hit a "not yet implemented" TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp rmod.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp pm2.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp smU.f90 -o - | FileCheck smU.f90

!--- rmod.f90
module rmod
  type :: t2
    integer :: val = 0
  end type
  interface operator(.plusb.)
    module procedure add2
  end interface
  !$omp declare reduction(.plusb.:t2:omp_out%val=omp_out%val+omp_in%val) &
  !$omp   initializer(omp_priv=t2(0))
contains
  type(t2) function add2(a, b)
    type(t2), intent(in) :: a, b
    add2%val = a%val + b%val
  end function
end module

!--- pm2.f90
module pm2
  use rmod, only: t2
  interface
    module subroutine w(y)
      import :: t2
      type(t2), intent(inout) :: y
    end subroutine
  end interface
end module

!--- smU.f90
! The imported operator reduction is materialized on demand for the clause inside
! the submodule, named from rmod's scope and source ultimate (op.plusb.), and the
! clause binds that same op; no TODO aborts the compile.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.plusb\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
submodule (pm2) smU
  use rmod, only: operator(.plusb.)
contains
  module subroutine w(y)
    type(t2), intent(inout) :: y
    integer :: i
    !$omp parallel do reduction(.plusb.:y)
    do i = 1, 10
      y%val = y%val + 1
    end do
  end subroutine
end submodule
