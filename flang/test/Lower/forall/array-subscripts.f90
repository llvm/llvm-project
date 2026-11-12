! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

  ! Make sure we use array values for subscripts that are arrays on the lhs so
  ! that copy-in/copy-out works correctly.
  integer :: a(4,4)
  forall(i=1:4,j=1:4)
    a(a(i,j),a(j,i)) = j - i*100
  end forall
end

! CHECK-LABEL: func.func @_QQmain() {
! CHECK: %[[A_ADDR:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<4x4xi32>>
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ADDR]]
! CHECK: hlfir.forall lb {
! CHECK: hlfir.yield %{{.*}} : i32
! CHECK: } ub {
! CHECK: hlfir.yield %{{.*}} : i32
! CHECK: }  (%[[I_ARG:.*]]: i32) {
! CHECK:   %[[I_REF:.*]] = hlfir.forall_index "i" %[[I_ARG]]
! CHECK:   hlfir.forall lb {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK:   } ub {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK:   }  (%[[J_ARG:.*]]: i32) {
! CHECK:     %[[J_REF:.*]] = hlfir.forall_index "j" %[[J_ARG]]
! CHECK:     hlfir.region_assign {
! CHECK:       hlfir.yield %{{.*}} : i32
! CHECK:     } to {
! CHECK:       hlfir.designate %[[A]]#0 ({{.*}}, {{.*}})
! CHECK:       hlfir.yield %{{.*}} : !fir.ref<i32>
! CHECK:     }
! CHECK:   }
! CHECK: }
