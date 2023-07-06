! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: }

program acc_private
  integer :: i, c
  integer, parameter :: n = 100
  real, dimension(n) :: a, b

! CHECK: %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<100xf32>>
! CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}

  !$acc loop private(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[C]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "c"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[C_PRIVATE]] : !fir.ref<i32>)
! CHECK: acc.yield

  !$acc loop private(b)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %{{.*}}, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.loop private(@privatization_ref_100xf32 -> %[[B_PRIVATE]] : !fir.ref<!fir.array<100xf32>>) {

end program
