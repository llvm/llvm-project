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

! CHECK: %[[C8:.*]] = arith.constant 8 : index

! CHECK: fir.do_concurrent.loop {{.*}} {
! CHECK:   %[[LOCAL_ALLOC:.*]] = fir.alloca !fir.array<8xf32> {bindc_name = "a", pinned, uniq_name = "{{.*}}local_assocEa"}
! CHECK:   %[[LOCAL_SHAPE:.*]] = fir.shape %[[C8]] :
! CHECK:   %[[LOCAL_DECL:.*]]:2 = hlfir.declare %[[LOCAL_ALLOC]](%[[LOCAL_SHAPE]])
! CHECK:   hlfir.designate %[[LOCAL_DECL]]#0 (%{{.*}})
! CHECK: }
