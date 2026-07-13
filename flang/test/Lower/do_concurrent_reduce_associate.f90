! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

subroutine dc_associate_reduce
  integer :: i
  real, allocatable, dimension(:) :: x

  associate(x_associate => x)
  do concurrent (i = 1:10) reduce(+: x_associate)
  end do
  end associate
end subroutine

! CHECK-LABEL: func.func @_QPdc_associate_reduce() {
! CHECK:         %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:         %[[ASSOC_DECL:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "{{.*}}x_associate"}
! CHECK:         fir.store %[[ASSOC_DECL]]#0 to %[[BOX_ALLOC]]
! CHECK-NEXT:    fir.do_concurrent {
! CHECK:           fir.do_concurrent.loop {{.*}} reduce(byref @{{.*}} #fir.reduce_attr<add> %[[BOX_ALLOC]] -> %{{.*}} : !{{.*}}) {
! CHECK:         }
! CHECK:       }
