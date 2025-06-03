! RUN: %flang_fc1 -emit-hlfir -mmlir --enable-delayed-privatization-staging=true -o - %s | FileCheck %s

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

! CHECK: fir.local {type = local} @[[LOCALIZER:.*local_assocEa.*]] : !fir.box<!fir.array<8xf32>> init {
! CHECK-NEXT: ^{{.*}}(%{{.*}}: !{{.*}}, %[[LOCAL_ARG:.*]]: !{{.*}}):
! CHECK-NEXT:   %[[C8:.*]] = arith.constant 8 : index
! CHECK-NEXT:   %[[SHAPE:.*]] = fir.shape %[[C8]]
! CHECK-NEXT:   %[[TMP_ALLOC:.*]] = fir.allocmem !{{.*}} {bindc_name = ".tmp", {{.*}}}
! CHECK:        %[[TMP_DECL:.*]]:2 = hlfir.declare %[[TMP_ALLOC]](%[[SHAPE]])
! CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
! CHECK-NEXT:   %[[C8:.*]] = arith.constant 8 : index
! CHECK-NEXT:   %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C1]], %[[C8]]
! CHECK-NEXT:   %[[TMP_BOX:.*]] = fir.embox %[[TMP_DECL]]#0(%[[SHAPE_SHIFT]])
! CHECK-NEXT:   fir.store %[[TMP_BOX]] to %[[LOCAL_ARG]]
! CHECK-NEXT:   fir.yield(%[[LOCAL_ARG]] : !fir.ref<!fir.box<!fir.array<8xf32>>>)
! CHECK-NEXT: }

! CHECK: fir.do_concurrent.loop {{.*}} local(@[[LOCALIZER]] %{{.*}} -> %[[LOCAL_ARG:.*]] : {{.*}}) {
! CHECK:   %[[LOCAL_DECL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]]
! CHECK:   %[[LOCAL_LD:.*]] = fir.load %[[LOCAL_DECL]]#0 : !fir.ref<!fir.box<!fir.array<8xf32>>>
! CHECK:   hlfir.designate %[[LOCAL_LD]] (%{{.*}})
! CHECK: }
