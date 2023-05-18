! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! Check that assumed shape lower bounds are applied before passing the
! descriptor to the runtime call.

  real, target :: a(10:20,99:100)
  call s2(a,17,-100)
contains
  subroutine show(bounds)
    integer(8) :: bounds(:)
    print *, bounds
  end subroutine
  subroutine s2(a,n,n2)
    Real a(n:,n2:)
    call show(ubound(a, kind=8))
  End Subroutine
end

! CHECK-LABEL: func.func private @_QFPs2
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>>
! CHECK: %[[BOX:.*]] = fir.rebox %[[ARG0]](%{{.*}}) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAUbound(%{{.*}}, %[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
