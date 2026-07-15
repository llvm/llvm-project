! Hermetic module files (-fhermetic-module-files) embed a module's USEd modules
! into its own .mod. This exercises an imported operator declare reduction that
! lives in such an embedded module: the wrapper herm_wrap re-exports herm_base
! (which owns the reduction) and is compiled hermetically, then herm_base.mod is
! removed so the consumer can only reach the reduction through the embedding.
! Materializing it requires (a) the embedded module's parse tree to stay live at
! lowering time, and (b) the reduction's module file to round-trip (its internal
! symbol is not written as an invalid use-only item). Runtime verified out of
! tree (runs to 100). https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp herm_base.f90
! RUN: %flang_fc1 -fhermetic-module-files -fsyntax-only -fopenmp herm_wrap.f90
! RUN: rm herm_base.mod
! RUN: %flang_fc1 -emit-hlfir -fopenmp herm.use.f90 -o - | FileCheck herm.use.f90

!--- herm_base.f90
module herm_base
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

!--- herm_wrap.f90
module herm_wrap
  use herm_base
end module

!--- herm.use.f90
! The reduction lives in the embedded herm_base; materialization must reach it
! through the hermetic herm_wrap.mod and name it from its source module.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*base[A-Za-z0-9_.]*op\.remote\.[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program main
  use herm_wrap, only: t, operator(.local.) => operator(.remote.)
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(.local.:x)
  do i = 1, 100
    x%val = x%val + 1
  end do
  print *, x%val
end program
