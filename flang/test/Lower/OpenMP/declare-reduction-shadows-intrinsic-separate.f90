! Separate-compilation counterpart of declare-reduction-shadows-intrinsic-use-assoc:
! a user declare reduction whose name shadows an intrinsic (max) is declared in a
! module compiled on its own to a .mod, and a separate consumer USEs the module
! and reduces with the shadowing name. The consumer is a distinct translation unit
! that never lowers the module's own declaration, so the clause (the
! shadowing-intrinsic bind site) must lazily materialize the imported op rather
! than hit a "not yet implemented" TODO.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: rm -rf %t && split-file %s %t && cd %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp shadow.mod.f90
! RUN: %flang_fc1 -emit-hlfir -fopenmp shadow.use.f90 -o - | FileCheck shadow.use.f90

!--- shadow.mod.f90
module m
  type :: t
    integer :: v = 0
  end type
  !$omp declare reduction(max : t : omp_out%v = max(omp_out%v, omp_in%v)) &
  !$omp   initializer(omp_priv = t(-2147483647))
end module

!--- shadow.use.f90
! The op is owner-qualified to the module and carries the mangled shadowing name
! (op.max); the USE-associated clause materializes and binds that same op, so no
! TODO aborts the compile.
! CHECK: omp.declare_reduction @[[RED:_QQ[A-Za-z0-9_.]*op\.max[A-Za-z0-9_.]*]] : !fir.ref
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
! CHECK-NOT: not yet implemented
program p
  use m
  type(t) :: x
  integer :: i
  x = t(-2147483647)
  !$omp parallel do reduction(max : x)
  do i = 1, 10
    x%v = max(x%v, i)
  end do
  print *, x%v   ! expected 10
end program
