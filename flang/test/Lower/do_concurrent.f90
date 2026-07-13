! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Simple tests for structured concurrent loops with loop-control.

pure function bar(n, m)
   implicit none
   integer, intent(in) :: n, m
   integer :: bar
   bar = n + m
end function

!CHECK-LABEL: sub1
subroutine sub1(n)
   implicit none
   integer :: n, m, i, j, k
   integer, dimension(n) :: a
!CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFsub1En"}
!CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFsub1Ea"}

!CHECK: %[[LB1:.*]] = arith.constant 1 : i32
!CHECK: %[[LB1_CVT:.*]] = fir.convert %[[LB1]] : (i32) -> index
!CHECK: %[[UB1:.*]] = fir.load %{{.*}}#0 : !fir.ref<i32>
!CHECK: %[[UB1_CVT:.*]] = fir.convert %[[UB1]] : (i32) -> index

!CHECK: %[[LB2:.*]] = arith.constant 1 : i32
!CHECK: %[[LB2_CVT:.*]] = fir.convert %[[LB2]] : (i32) -> index
!CHECK: %[[UB2:.*]] = fir.call @_QPbar(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
!CHECK: %[[UB2_CVT:.*]] = fir.convert %[[UB2]] : (i32) -> index

!CHECK: %[[LB3:.*]] = arith.constant 5 : i32
!CHECK: %[[LB3_CVT:.*]] = fir.convert %[[LB3]] : (i32) -> index
!CHECK: %[[UB3:.*]] = arith.constant 10 : i32
!CHECK: %[[UB3_CVT:.*]] = fir.convert %[[UB3]] : (i32) -> index

!CHECK: fir.do_concurrent
!CHECK:   %[[I:.*]] = fir.alloca i32 {bindc_name = "i"}
!CHECK:   %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
!CHECK:   %[[J:.*]] = fir.alloca i32 {bindc_name = "j"}
!CHECK:   %[[J_DECL:.*]]:2 = hlfir.declare %[[J]]
!CHECK:   %[[K:.*]] = fir.alloca i32 {bindc_name = "k"}
!CHECK:   %[[K_DECL:.*]]:2 = hlfir.declare %[[K]]

!CHECK:   fir.do_concurrent.loop (%[[I_IV:.*]], %[[J_IV:.*]], %[[K_IV:.*]]) =
!CHECK-SAME:                     (%[[LB1_CVT]], %[[LB2_CVT]], %[[LB3_CVT]]) to
!CHECK-SAME:                     (%[[UB1_CVT]], %[[UB2_CVT]], %[[UB3_CVT]]) step
!CHECK-SAME:                     (%{{.*}}, %{{.*}}, %{{.*}}) {
!CHECK:       %[[I_IV_CVT:.*]] = fir.convert %[[I_IV]] : (index) -> i32
!CHECK:       fir.store %[[I_IV_CVT]] to %[[I_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[J_IV_CVT:.*]] = fir.convert %[[J_IV]] : (index) -> i32
!CHECK:       fir.store %[[J_IV_CVT]] to %[[J_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[K_IV_CVT:.*]] = fir.convert %[[K_IV]] : (index) -> i32
!CHECK:       fir.store %[[K_IV_CVT]] to %[[K_DECL]]#0 : !fir.ref<i32>

!CHECK:       %[[N_VAL:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[I_VAL:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
!CHECK:       %[[I_VAL_CVT:.*]] = fir.convert %[[I_VAL]] : (i32) -> i64
!CHECK:       %[[A_ELEM:.*]] = hlfir.designate %[[A_DECL]]#0 (%[[I_VAL_CVT]])
!CHECK:       hlfir.assign %[[N_VAL]] to %[[A_ELEM]] : i32, !fir.ref<i32>
   do concurrent(i=1:n, j=1:bar(n*m, n/m), k=5:10)
      a(i) = n
   end do
end subroutine

!CHECK-LABEL: sub2
subroutine sub2(n)
   implicit none
   integer :: n, m, i, j
   integer, dimension(n) :: a
!CHECK: %[[LB1:.*]] = arith.constant 1 : i32
!CHECK: %[[LB1_CVT:.*]] = fir.convert %[[LB1]] : (i32) -> index
!CHECK: %[[UB1:.*]] = fir.load %{{.*}}#0 : !fir.ref<i32>
!CHECK: %[[UB1_CVT:.*]] = fir.convert %[[UB1]] : (i32) -> index
!CHECK: fir.do_concurrent
!CHECK:   fir.do_concurrent.loop (%{{.*}}) = (%[[LB1_CVT]]) to (%[[UB1_CVT]]) step (%{{.*}})

!CHECK: %[[LB2:.*]] = arith.constant 1 : i32
!CHECK: %[[LB2_CVT:.*]] = fir.convert %[[LB2]] : (i32) -> index
!CHECK: %[[UB2:.*]] = fir.call @_QPbar(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
!CHECK: %[[UB2_CVT:.*]] = fir.convert %[[UB2]] : (i32) -> index
!CHECK: fir.do_concurrent
!CHECK:   fir.do_concurrent.loop (%{{.*}}) = (%[[LB2_CVT]]) to (%[[UB2_CVT]]) step (%{{.*}})
   do concurrent(i=1:n)
      do concurrent(j=1:bar(n*m, n/m))
         a(i) = n
      end do
   end do
end subroutine

!CHECK-LABEL: unstructured
subroutine unstructured(inner_step)
  integer(4) :: i, j, inner_step

!CHECK-NOT: cf.br
!CHECK-NOT: cf.cond_br
!CHECK:     %[[LB1:.*]] = arith.constant 1 : i32
!CHECK:     %[[LB1_CVT:.*]] = fir.convert %c1_i32 : (i32) -> i16
!CHECK:     %[[UB1:.*]] = arith.constant 5 : i32
!CHECK:     %[[UB1_CVT:.*]] = fir.convert %c5_i32 : (i32) -> i16
!CHECK:     %[[STP1:.*]] = arith.constant 1 : i16

!CHECK-NOT: cf.br
!CHECK-NOT: cf.cond_br
!CHECK:     %[[LB2:.*]] = arith.constant 3 : i32
!CHECK:     %[[LB2_CVT:.*]] = fir.convert %[[LB2]] : (i32) -> i16
!CHECK:     %[[UB2:.*]] = arith.constant 9 : i32
!CHECK:     %[[UB2_CVT:.*]] = fir.convert %[[UB2]] : (i32) -> i16
!CHECK:     %[[STP2:.*]] = fir.load %{{.*}}#0 : !fir.ref<i32>
!CHECK:     %[[STP2_CVT:.*]] = fir.convert %[[STP2]] : (i32) -> i16
!CHECK:     fir.store %[[STP2_CVT]] to %{{.*}} : !fir.ref<i16>
!CHECK:     cf.br ^[[I_LOOP_HEADER:.*]]

!CHECK: ^[[I_LOOP_HEADER]]:
!CHECK-NEXT: %{{.*}} = fir.load %{{.*}} : !fir.ref<i16>
!CHECK-NEXT: %{{.*}} = arith.constant 0 : i16
!CHECK-NEXT: %{{.*}} = arith.cmpi sgt, %{{.*}}, %{{.*}}: i16
!CHECK-NEXT: cf.cond_br %{{.*}}, ^[[J_LOOP_HEADER:.*]], ^{{.*}}

!CHECK: ^[[J_LOOP_HEADER]]:
!CHECK-NEXT: %[[RANGE:.*]] = arith.subi %[[UB2_CVT]], %[[LB2_CVT]] : i16
!CHECK-NEXT: %{{.*}} = arith.addi %[[RANGE]], %[[STP2_CVT]] : i16
!CHECK-NEXT: %{{.*}} = arith.divsi %{{.*}}, %[[STP2_CVT]] : i16
  do concurrent (integer(2)::i=1:5, j=3:9:inner_step, i.ne.3)
    goto (7, 7) i+1
    print*, 'E:', i, j
  7 continue
  enddo
end subroutine unstructured
