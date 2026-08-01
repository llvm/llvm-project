! A user-defined declare reduction whose name shadows an intrinsic
! (max/min/iand/ior/ieor), declared in a module and used through USE
! association, must lower cross-scope like the named path: the op is named from
! the module scope and the source ultimate (op.max), and the use-site clause
! binds that same op. This used to be a clean TODO ("not yet supported for
! imported or renamed reductions"); the imported case is now handled by naming
! from the found symbol's ultimate, and the renamed case is rejected by
! semantics before lowering (the renamed name resolves to the intrinsic, not the
! user reduction). Reproducer runs to 10 (repro/xmod-shadow/xsh.f90).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t
    integer :: v = 0
  end type
  !$omp declare reduction(max : t : omp_out%v = max(omp_out%v, omp_in%v)) &
  !$omp   initializer(omp_priv = t(-2147483647))
end module

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

! The op is owner-qualified to the module (_QQMm) and carries the mangled
! shadowing name (op.max). The USE-associated clause binds the same op.
! CHECK: omp.declare_reduction @[[RED:_QQMmop.max_byref_rec__QMmTt]] : !fir.ref<!fir.type<{{[^>]*}}t{v:i32}>>
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
