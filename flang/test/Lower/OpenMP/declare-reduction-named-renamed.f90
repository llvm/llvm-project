! A named user-defined declare reduction imported under a rename
! (myadd_alias => myadd) must lower to the source module's op, not the local
! alias spelling. The reduction op is named from the module scope and the source
! ultimate name (myadd), and the use-site clause (which spells the alias) binds
! that same source op. This is the named analogue of
! declare-reduction-operator-renamed.f90; the shared owner-qualified naming
! helper resolves the alias to the source by taking the symbol's ultimate.
! Reproducer runs to 55 (repro/xmod-named/xmn.f90).

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module m
  type :: t
    integer :: v = 0
  end type
  !$omp declare reduction(myadd : t : omp_out%v = omp_out%v + omp_in%v) &
  !$omp   initializer(omp_priv = t(0))
end module

program p
  use m, only : t, myadd_alias => myadd
  type(t) :: x
  integer :: i
  x = t(0)
  !$omp parallel do reduction(myadd_alias : x)
  do i = 1, 10
    x%v = x%v + i
  end do
  print *, x%v   ! expected 55
end program

! The op is named from the module owner (_QQMm) and the source name (myadd),
! never the alias (myadd_alias). The clause spelled with the alias binds the
! same source op.
! CHECK: omp.declare_reduction @[[RED:_QQMmmyadd_byref_rec__QMmTt]] : !fir.ref<!fir.type<{{[^>]*}}t{v:i32}>>
! CHECK-NOT: myadd_alias
! CHECK: omp.wsloop
! CHECK-SAME: reduction(byref @[[RED]]
