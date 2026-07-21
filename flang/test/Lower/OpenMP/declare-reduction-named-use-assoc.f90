! A named user-defined declare reduction declared in a module and used in a
! separate program unit must lower correctly cross-scope: the reduction op is
! named from the module scope (owner-qualified) and the use-site clause binds
! that same op. Lazy clause-driven materialization emits the imported op for
! separate compilation; here the module and program share a file, but the naming
! contract is the same. Single-type reductions are named without a
! type suffix, so this locks the un-suffixed cross-scope spelling.
! https://github.com/llvm/llvm-project/issues/207255

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t
    integer :: v = 0
  end type
  !$omp declare reduction(myadd: t: omp_out%v = omp_out%v + omp_in%v) &
  !$omp   initializer(omp_priv=t(0))
end module

program main
  use m
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(myadd:x)
  do i = 1, 5
    x%v = x%v + i
  end do
  print *, x%v
end program

! The op is owner-qualified to the module scope (_QQMm...) and carries the
! type/byref suffix. The use-site clause binds the same op.
! CHECK: omp.declare_reduction @[[RED:_QQMmmyadd_byref_rec__QMmTt]] : !fir.ref<!fir.type<{{[^>]*}}t{v:i32}>>
! CHECK-NOT: omp.declare_reduction @{{.*}}myadd{{.*}}myadd
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
