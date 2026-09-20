! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine local_assoc
  implicit none
  integer  i
  real, dimension(2:11) :: aa

  associate(a => aa(4:))
    do concurrent (i = 4:11) local(a)
      a(i) = 0
    end do
  end associate
end subroutine local_assoc

! A constant-shape, trivial-element local is localized unboxed as a plain
! fir.array with no init region. The end-of-line anchor confirms there is no
! `init {` region.
! CHECK: fir.local {type = local} @[[LOCALIZER:.*local_assocEa.*]] : !fir.array<8xf32>{{$}}

! CHECK: func.func @_QPlocal_assoc()
! CHECK: %[[ASSOC_DECL:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "{{.*}}local_assocEa"}
! CHECK: fir.do_concurrent.loop {{.*}} local(@[[LOCALIZER]] %[[ASSOC_DECL]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<!fir.array<8xf32>>) {
! CHECK:   %[[LOCAL_DECL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]](%{{.*}})
! CHECK:   hlfir.designate %[[LOCAL_DECL]]#0 (%{{.*}})
! CHECK: }
